"""Validated MiniMax H3 optimization profile for NVIDIA V100 / SM70."""
import logging
import torch
from .native_dynamic_vbar import NativeDynamicVBARPolicy, CONTROLLER_KEY
from .weight_profile import FP8_E4M3_PROFILE, INT8_CONVROT_PROFILE, WEIGHT_PROFILE_OPTION_KEY, detect_h3_weight_profile
from .phase_model_release import PriorStageDynamicModelReleaser, install_load_phase_guard, mark_phase_managed
from .flash_attention import V100FlashAttention
from .h3_mixed_precision import H3V100MixedPrecision, PROJECTION_WEIGHT_REUSE_OPTION_KEY
from .h3_prefetch_guard import H3RuntimePrefetchGuard
from .sol_attention import MODE_FLASH, MODE_SOL
from .tokenwise_chunking import H3TokenwiseMLPChunking
from .fused_sol_speed import patch_model_for_fused_sol_speed
from .sol_adaptive_policy import POLICY_KEY as ADAPTIVE_POLICY_KEY, install_adaptive_budget
from .h3_easycache import MODE_OFF as EASYCACHE_OFF, SUPPORTED_MODES as EASYCACHE_MODES, patch_model_for_h3_easycache
from .runtime_memory import install_runtime_memory_policy, patch_model_for_runtime_memory_lifecycle
LOGGER = logging.getLogger('H3V100Optimize')
SOURCE_MODEL_FINGERPRINT_KEY = 'v100_h3_source_model_fingerprint'
FLASH_MIN_TOKENS = 1024
SOL_SPEED_MIN_TOKENS = 16384
SOL_TOPK_TAIL_BLOCKS = 32
SOL_MINIMUM_GAIN_PERCENT = 5.0
SOL_ROUTE_MEMORY_LIMIT_MIB = 1024
QUALITY_PROFILE = 'quality'
SPEED_PROFILE = 'speed'
ULTRA_PROFILE = 'ultra'
MANUAL_PROFILE = 'manual'
QUALITY_PROFILES = (QUALITY_PROFILE, SPEED_PROFILE, MANUAL_PROFILE)
SOL_QUALITY_PROFILES = (QUALITY_PROFILE, SPEED_PROFILE, ULTRA_PROFILE, MANUAL_PROFILE)
SOL_SPEED_TAU = 2.0
SOL_ULTRA_TAU = 2.5
EASYCACHE_QUALITY_VIDEO_THRESHOLD = 0.15
EASYCACHE_QUALITY_VIDEO_FRAME_P95_THRESHOLD = 0.18
EASYCACHE_QUALITY_AUDIO_THRESHOLD = 0.1
EASYCACHE_SPEED_VIDEO_THRESHOLD = 0.2
EASYCACHE_SPEED_VIDEO_FRAME_P95_THRESHOLD = 0.25
EASYCACHE_SPEED_AUDIO_THRESHOLD = 0.2
MLP_CHUNK_TOKENS = 640
CACHE_TRIM_THRESHOLD_MIB = 2048
FP16_PROJECTION_PATHS = (('attn', 'qkv_proj'), ('attn', 'out_proj'), ('mlp', 'fc1'), ('mlp', 'fc2'))

def _diffusion_model(model):
    try:
        return model.get_model_object('diffusion_model')
    except Exception:
        return None

def _is_h3(diffusion_model):
    model_type = type(diffusion_model)
    if not (model_type.__module__ == 'comfy.ldm.minimax.model' and model_type.__name__ == 'MiniMaxH3Model' and (getattr(diffusion_model, 'hidden_size', None) == 5376) and hasattr(diffusion_model, 'token_refiner') and hasattr(diffusion_model, 'rope') and hasattr(diffusion_model, 'video_patch_proj') and hasattr(diffusion_model, 'audio_patch_proj')):
        return False
    blocks = getattr(diffusion_model, 'blocks', None)
    if not blocks or len(blocks) != 50:
        return False
    for block in blocks:
        attention = getattr(block, 'attn', None)
        if not (attention is not None and getattr(attention, 'head_dim', None) == 128 and (getattr(attention, 'heads', None) == 56) and all((hasattr(attention, name) for name in ('qkv_proj', 'q_norm', 'k_norm', 'out_proj', 'heads')))):
            return False
    return True

def _source_model_fingerprint(model, weight_profile=None):
    """Stable across optimization clones, different across LoRA patch sets."""
    clone_base_uuid = getattr(model, 'clone_base_uuid', None)
    patches_uuid = getattr(model, 'patches_uuid', None)
    return (str(clone_base_uuid) if clone_base_uuid is not None else f'id:{id(model)}', str(patches_uuid) if patches_uuid is not None else 'unpatched', str(weight_profile) if weight_profile is not None else 'unknown-weight-profile')

def _force_dynamic_projection_casts(model):
    """Force compressed projections into the validated compute-precision path."""
    patched = model.clone()
    for index in range(50):
        for owner, projection in FP16_PROJECTION_PATHS:
            patched.add_object_patch(f'diffusion_model.blocks.{index}.{owner}.{projection}.comfy_force_cast_weights', True)
    return patched

def _resolve_sol_profile(profile, *, tau, start_percent, end_percent):
    """Resolve Sol controls without changing serialized expert workflows."""
    if profile not in SOL_QUALITY_PROFILES:
        raise ValueError(f'sol_quality_profile must be one of {SOL_QUALITY_PROFILES!r}.')
    resolved_tau = float(tau)
    if profile == SPEED_PROFILE:
        resolved_tau = max(resolved_tau, SOL_SPEED_TAU)
    elif profile == ULTRA_PROFILE:
        resolved_tau = max(resolved_tau, SOL_ULTRA_TAU)
    return (resolved_tau, float(start_percent), float(end_percent))

def _resolve_easycache_profile(profile, *, video_threshold, video_frame_p95_threshold, audio_threshold):
    """Resolve EasyCache gates without weakening serialized workflows."""
    if profile not in QUALITY_PROFILES:
        raise ValueError(f'easycache_quality_profile must be one of {QUALITY_PROFILES!r}.')
    resolved = (float(video_threshold), float(video_frame_p95_threshold), float(audio_threshold))
    if profile == SPEED_PROFILE:
        resolved = (max(resolved[0], EASYCACHE_SPEED_VIDEO_THRESHOLD), max(resolved[1], EASYCACHE_SPEED_VIDEO_FRAME_P95_THRESHOLD), max(resolved[2], EASYCACHE_SPEED_AUDIO_THRESHOLD))
    return resolved

def _dual_gpu_secondary_options():
    from .dual_attention import available_secondary_device_options
    return available_secondary_device_options()

def _patch_model_for_dual_attention(model, secondary_device):
    from .dual_attention import patch_model_for_dual_attention
    return patch_model_for_dual_attention(model, secondary_device=secondary_device)

class H3V100Optimize:
    """Apply the frozen H3 V100 precision, attention, and memory policy."""

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'attention_backend': (('Flash', 'SOL'), {'default': 'Flash'}), 'sol_quality_profile': (SOL_QUALITY_PROFILES, {'default': QUALITY_PROFILE}), 'sol_tau': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 4.0, 'step': 0.05}), 'easycache_mode': (('Off', 'Quality', 'Speed'), {'default': 'Off'}), 'dual_gpu': ('BOOLEAN', {'default': False}), 'dual_gpu_secondary': (_dual_gpu_secondary_options(), {'default': 'auto'})}}
    RETURN_TYPES = ('MODEL',)
    RETURN_NAMES = ('model',)
    FUNCTION = 'patch'
    CATEGORY = 'MiniMax H3/V100'
    DESCRIPTION = 'MiniMax H3 for V100/SM70: exact Flash or calibrated sparse Sol, optional audiovisual EasyCache, and optional second V100. INT8 ConvRot and scaled FP8 E4M3. Connect this MODEL to both samplers for split sampling. Single-card Sol enables Adaptive Budget; dual-card Sol uses the base route.'

    def patch(self, model, mixed_precision=True, attention_backend=MODE_FLASH, sol_tau=1.0, sol_start_percent=0.2, sol_end_percent=0.8, long_projection_reuse=True, sol_topk_tail_blocks=SOL_TOPK_TAIL_BLOCKS, sol_min_tokens=SOL_SPEED_MIN_TOKENS, sol_minimum_gain_percent=SOL_MINIMUM_GAIN_PERCENT, sol_route_memory_limit_mib=SOL_ROUTE_MEMORY_LIMIT_MIB, easycache_mode=False, easycache_video_threshold=EASYCACHE_QUALITY_VIDEO_THRESHOLD, easycache_video_frame_p95_threshold=EASYCACHE_QUALITY_VIDEO_FRAME_P95_THRESHOLD, easycache_audio_threshold=EASYCACHE_QUALITY_AUDIO_THRESHOLD, easycache_start_percent=0.15, easycache_end_percent=0.85, easycache_video_storage='int8', easycache_memory_limit_mib=512, sol_quality_profile=QUALITY_PROFILE, easycache_quality_profile=QUALITY_PROFILE, sol_adaptive_budget=True, dual_gpu=False, dual_gpu_secondary='auto', cold_start_prefetch='off'):
        attention_backend = {'Flash': MODE_FLASH, 'SOL': MODE_SOL}.get(attention_backend, attention_backend)
        if easycache_mode in ('Quality', 'Speed'):
            easycache_quality_profile = easycache_mode.lower()
            easycache_video_threshold = EASYCACHE_QUALITY_VIDEO_THRESHOLD
            easycache_video_frame_p95_threshold = EASYCACHE_QUALITY_VIDEO_FRAME_P95_THRESHOLD
            easycache_audio_threshold = EASYCACHE_QUALITY_AUDIO_THRESHOLD
        easycache_mode = _enabled_easycache(easycache_mode)
        if CONTROLLER_KEY in model.model_options.get('transformer_options', {}):
            raise RuntimeError('H3 V100 Optimize is already installed on this MODEL. Use one Optimize node per model branch; connect its output to both samplers for split sampling. To compare settings, branch from the model before Optimize.')
        if not bool(mixed_precision):
            LOGGER.warning('H3 V100 Optimize ignored legacy mixed_precision=False; the validated stable precision contract is always enabled.')
        mixed_precision = True
        if cold_start_prefetch != 'off':
            LOGGER.warning('H3 ignored retired cold_start_prefetch=%r; using the validated native demand path.', cold_start_prefetch)
        if attention_backend not in (MODE_FLASH, MODE_SOL):
            raise ValueError(f'attention_backend must be {MODE_FLASH!r} or {MODE_SOL!r}; received {attention_backend!r}.')
        fp16_mlp = True
        if not 0.0 <= float(sol_tau) <= 4.0:
            raise ValueError('sol_tau must be between 0 and 4.')
        if not 0.0 <= float(sol_minimum_gain_percent) <= 50.0:
            raise ValueError('sol_minimum_gain_percent must be between 0 and 50.')
        if easycache_mode not in EASYCACHE_MODES:
            raise ValueError(f'easycache_mode must be one of {EASYCACHE_MODES!r}.')
        effective_sol_tau, effective_sol_start_percent, effective_sol_end_percent = _resolve_sol_profile(str(sol_quality_profile), tau=sol_tau, start_percent=sol_start_percent, end_percent=sol_end_percent)
        effective_easycache_video_threshold, effective_easycache_video_frame_p95_threshold, effective_easycache_audio_threshold = _resolve_easycache_profile(str(easycache_quality_profile), video_threshold=easycache_video_threshold, video_frame_p95_threshold=easycache_video_frame_p95_threshold, audio_threshold=easycache_audio_threshold)
        if not 0.0 <= effective_sol_start_percent < effective_sol_end_percent <= 1.0:
            raise ValueError('Sol manual mode requires 0 <= start < end <= 1.')
        if not 0.0 <= float(easycache_video_threshold) <= 1.0:
            raise ValueError('easycache_video_threshold must be between 0 and 1.')
        if not 0.0 <= float(easycache_video_frame_p95_threshold) <= 4.0:
            raise ValueError('easycache_video_frame_p95_threshold must be between 0 and 4.')
        if not 0.0 <= float(easycache_audio_threshold) <= 1.0:
            raise ValueError('easycache_audio_threshold must be between 0 and 1.')
        if not 0.0 <= float(easycache_start_percent) < float(easycache_end_percent) <= 1.0:
            raise ValueError('EasyCache requires 0 <= start_percent < end_percent <= 1.')
        if not torch.cuda.is_available() or not any((torch.cuda.get_device_capability(index) == (7, 0) for index in range(torch.cuda.device_count()))):
            raise RuntimeError('H3 V100 Optimize requires an SM70 CUDA device.')
        diffusion_model = _diffusion_model(model)
        if not _is_h3(diffusion_model):
            model_type = type(diffusion_model)
            raise RuntimeError(f'H3 V100 Optimize accepts only the validated MiniMax H3 model; received {model_type.__module__}.{model_type.__name__}.')
        weight_profile = detect_h3_weight_profile(diffusion_model)
        if bool(dual_gpu):
            choices = _dual_gpu_secondary_options()
            if len(choices) <= 1:
                raise RuntimeError('Dual GPU requires another visible V100/SM70 device in the current ComfyUI process.')
            if str(dual_gpu_secondary) not in choices:
                raise RuntimeError(f'dual_gpu_secondary is not a compatible visible V100: {dual_gpu_secondary!r}; available={choices!r}.')
            if weight_profile not in (FP8_E4M3_PROFILE, INT8_CONVROT_PROFILE):
                raise RuntimeError(f'Dual GPU requires a validated FP8 E4M3 scaled or INT8-ConvRot H3 core; received {weight_profile!r}.')
        configured = model.clone()
        is_dynamic = getattr(configured, 'is_dynamic', None)
        if not callable(is_dynamic) or not bool(is_dynamic()):
            raise RuntimeError('H3 V100 requires ComfyUI DynamicVRAM. Remove --disable-dynamic-vram and --lowvram, restart ComfyUI, and reload the model.')
        configured.set_model_compute_dtype(torch.float16)
        configured = _force_dynamic_projection_casts(configured)
        configured_options = configured.model_options.setdefault('transformer_options', {})
        inherited_adaptive = configured_options.get(ADAPTIVE_POLICY_KEY)
        if isinstance(inherited_adaptive, dict) and inherited_adaptive.get('owner') == 'main':
            configured_options.pop(ADAPTIVE_POLICY_KEY)
        dynamic_vbar_controller = NativeDynamicVBARPolicy(0.0, base_model=getattr(configured, 'model', None))
        configured_options[CONTROLLER_KEY] = dynamic_vbar_controller
        configured_options[WEIGHT_PROFILE_OPTION_KEY] = weight_profile
        configured_options[SOURCE_MODEL_FINGERPRINT_KEY] = _source_model_fingerprint(model, weight_profile)
        configured_options[PROJECTION_WEIGHT_REUSE_OPTION_KEY] = True
        install_runtime_memory_policy(configured_options, soft_free_floor_mib=CACHE_TRIM_THRESHOLD_MIB, hard_free_floor_mib=512, minimum_reclaimable_mib=512, cooldown_checks=12)
        optimized, = H3V100MixedPrecision().patch(configured, enabled=True, v100_only=True)
        optimized, = H3TokenwiseMLPChunking().patch(optimized, adaptive=True, chunk_tokens=MLP_CHUNK_TOKENS, cache_trim=True, cache_trim_threshold_mb=CACHE_TRIM_THRESHOLD_MIB, experimental_fp16=fp16_mlp, scaled_fp16_swiglu=True, dynamic_vbar_controller=dynamic_vbar_controller)
        optimized, = V100FlashAttention().patch(optimized, enabled=True, min_tokens=FLASH_MIN_TOKENS, attention_mode=MODE_FLASH, allow_sol=False)
        optimized = optimized.clone()
        transformer_options = optimized.model_options.setdefault('transformer_options', {})
        transformer_options['prefetch_dynamic_vbars'] = False
        transformer_options[PROJECTION_WEIGHT_REUSE_OPTION_KEY] = True
        transformer_options['v100_h3_selected_attention_backend'] = attention_backend
        transformer_options['v100_h3_sol_quality_profile'] = str(sol_quality_profile)
        transformer_options['v100_h3_easycache_quality_profile'] = str(easycache_quality_profile)
        install_load_phase_guard()
        phase_model_releaser = PriorStageDynamicModelReleaser()
        optimized, = H3RuntimePrefetchGuard().patch(optimized, enabled=False, adaptive_memory=True, experimental_fp16=fp16_mlp, projection_weight_reuse=True, phase_model_releaser=phase_model_releaser)
        if attention_backend == MODE_SOL:
            optimized, _sol_state = patch_model_for_fused_sol_speed(optimized, tau=effective_sol_tau, min_tokens=int(sol_min_tokens), minimum_gain_percent=float(sol_minimum_gain_percent), start_percent=effective_sol_start_percent, end_percent=effective_sol_end_percent, memory_limit_mib=int(sol_route_memory_limit_mib), quality_profile=str(sol_quality_profile))
            if not bool(dual_gpu):
                install_adaptive_budget(optimized.model_options.setdefault('transformer_options', {}), min_tokens=int(sol_min_tokens))
        if easycache_mode != EASYCACHE_OFF:
            optimized, _easycache_state = patch_model_for_h3_easycache(optimized, mode=str(easycache_mode), refresh_backend=str(attention_backend), video_threshold=effective_easycache_video_threshold, video_frame_p95_threshold=effective_easycache_video_frame_p95_threshold, audio_threshold=effective_easycache_audio_threshold, start_percent=float(easycache_start_percent), end_percent=float(easycache_end_percent), video_storage=str(easycache_video_storage), memory_limit_mib=int(easycache_memory_limit_mib), quality_profile=str(easycache_quality_profile))
        optimized = patch_model_for_runtime_memory_lifecycle(optimized)
        optimized = mark_phase_managed(optimized)
        if bool(dual_gpu):
            optimized = _patch_model_for_dual_attention(optimized, str(dual_gpu_secondary))
        return (optimized,)

def _enabled_easycache(value):
    if value is True or value in ('active', 'Quality', 'Speed'):
        return 'active'
    if value is False or value in ('off', 'shadow', 'Off'):
        return 'off'
    raise ValueError('EasyCache must be Off, Quality or Speed')
