"""Validated compressed-weight profiles for the H3 V100 runtime."""
from __future__ import annotations
WEIGHT_PROFILE_OPTION_KEY = 'v100_h3_weight_profile'
INT8_CONVROT_PROFILE = 'int8_convrot'
FP8_E4M3_PROFILE = 'fp8_e4m3_scaled'
_BLOCK_COUNT = 50
_PROJECTIONS = (('attn', 'qkv_proj'), ('attn', 'out_proj'), ('mlp', 'fc1'), ('mlp', 'fc2'))

def _projection_modules(diffusion_model):
    blocks = getattr(diffusion_model, 'blocks', None)
    if blocks is None or len(blocks) != _BLOCK_COUNT:
        raise RuntimeError('H3 V100 compressed-weight admission requires exactly 50 blocks.')
    for block_index, block in enumerate(blocks):
        for owner_name, projection_name in _PROJECTIONS:
            owner = getattr(block, owner_name, None)
            projection = getattr(owner, projection_name, None)
            if projection is None:
                raise RuntimeError(f'H3 V100 compressed-weight admission is missing blocks.{block_index}.{owner_name}.{projection_name}.')
            yield (block_index, owner_name, projection_name, projection)

def detect_h3_weight_profile(diffusion_model) -> str:
    """Classify all 200 core projections without materializing weight data."""
    formats = set()
    projections = []
    for block_index, owner_name, projection_name, projection in _projection_modules(diffusion_model):
        path = f'blocks.{block_index}.{owner_name}.{projection_name}'
        quant_format = getattr(projection, 'quant_format', None)
        formats.add(quant_format)
        params = getattr(getattr(projection, 'weight', None), '_params', None)
        if params is None or getattr(params, 'scale', None) is None:
            raise RuntimeError(f'H3 V100 compressed-weight admission found no scale for {path}.')
        if getattr(projection, 'pre_quant_scale', None) is not None:
            raise RuntimeError(f'H3 V100 does not support pre_quant_scale on prepared-weight projections: {path}.')
        projections.append((path, params))
    if formats == {'float8_e4m3fn'}:
        return FP8_E4M3_PROFILE
    if formats == {'int8_tensorwise'}:
        for path, params in projections:
            if not bool(getattr(params, 'convrot', False)):
                raise RuntimeError(f'H3 V100 accepts INT8 core projections only with ConvRot: {path}.')
            if int(getattr(params, 'convrot_groupsize', 0)) != 256:
                raise RuntimeError(f'H3 V100 accepts INT8 ConvRot group size 256 only: {path}.')
        return INT8_CONVROT_PROFILE
    printable = sorted(('dense' if value is None else str(value) for value in formats))
    raise RuntimeError(f'H3 V100 supports only a uniform INT8-ConvRot or FP8 E4M3 scaled core-weight profile; received formats={printable}.')

def weight_profile_from_options(options) -> str:
    """Keep direct helper tests and old patched models on the frozen INT8 policy."""
    if not isinstance(options, dict):
        return INT8_CONVROT_PROFILE
    return str(options.get(WEIGHT_PROFILE_OPTION_KEY, INT8_CONVROT_PROFILE))

def qkv_native_reserve_bytes(options, *, extreme: bool) -> int:
    """Driver-visible reserve for source-page faulting and weight expansion.

    Actual cuda:0 probes of the installed FP8 E4M3 checkpoint measured about
    502 MiB driver growth for QKV->FP16.  Keep a 2x normal margin and a 3x
    extreme margin.  INT8-ConvRot retains the previously validated floors.
    """
    profile = weight_profile_from_options(options)
    if profile == FP8_E4M3_PROFILE:
        mib = 1536 if extreme else 1024
    else:
        mib = 4096 if extreme else 2048
    return mib * 1024 ** 2

def dual_attention_decode_extra_bytes(profile, *, heads=56, head_dim=128, hidden=5376):
    """Additional primary workspace beyond the dual planner's FP8 baseline.

    ConvRot may use the portable inverse: INT8 source plus two full FP32
    buffers. The existing QKV baseline includes source plus FP16 output (3
    bytes/element); cover the extra 6 bytes and Hadamard/scale metadata even
    when this machine can select a lower-memory native decoder. Shard math
    must start only after inverse rotation in the original full weight basis.
    """
    if profile == FP8_E4M3_PROFILE:
        return 0
    if profile != INT8_CONVROT_PROFILE:
        raise ValueError(f'Unsupported dual attention weight profile: {profile!r}')
    if min(heads, head_dim, hidden) <= 0:
        raise ValueError('Invalid dual attention geometry')
    qkv_elements = 3 * heads * head_dim * hidden
    return 6 * qkv_elements + 256 * 256 * 4 + 3 * heads * head_dim * 4
