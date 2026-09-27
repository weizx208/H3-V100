"""Release inactive DynamicVRAM models at the H3 sampling phase boundary."""
import logging
import threading
LOGGER = logging.getLogger('H3V100PhaseRelease')
MODEL_MARKER = '_h3_v100_phase_boundary_managed'
MODEL_ATTACHMENT = 'h3_v100_phase_boundary_managed'
LOAD_GUARD_MARKER = '_h3_v100_load_phase_guard'
LOAD_GUARD_ORIGINAL = '_h3_v100_load_phase_guard_original'
_install_lock = threading.Lock()
_managed_keys_lock = threading.Lock()
_managed_model_keys = set()
_PIN_SUBSETS = ('weights-loaded', 'patches-loaded', 'weights', 'patches')

def _size_from(model, method_name):
    method = getattr(model, method_name, None)
    if not callable(method):
        return 0
    value = method()
    return 0 if value is None else max(0, int(value))

def _dynamic_pin_state(model):
    inner = getattr(model, 'model', None)
    states = getattr(inner, 'dynamic_pins', None)
    device = getattr(model, 'load_device', None)
    if not isinstance(states, dict) or device not in states:
        raise RuntimeError('DynamicVRAM pin state is unavailable')
    return states[device]

def _registered_pin_bytes(model):
    pin_state = _dynamic_pin_state(model)
    total = 0
    for subset in _PIN_SUBSETS:
        state = pin_state.get(subset)
        if state is not None and len(state) > 3:
            total += max(0, int(state[3][0]))
    return total

def _needs_dynamic_release(model):
    if _size_from(model, 'loaded_size') > 0:
        return True
    try:
        return _registered_pin_bytes(model) > 0
    except RuntimeError:
        return True

def _cleanup_prefetch_queues():
    """Drop references that may still pin a DynamicVRAM VBAR range."""
    try:
        import comfy.model_prefetch as model_prefetch
    except ImportError:
        return False
    cleanup = getattr(model_prefetch, 'cleanup_prefetch_queues', None)
    if not callable(cleanup):
        return False
    cleanup()
    return True

def _release_dynamic_gpu_keep_unregistered_host(model, model_management):
    """Release inactive GPU residency/registration while preserving host data."""
    try:
        ram_before = _size_from(model, 'loaded_ram_size')
        pin_state = _dynamic_pin_state(model)
        if bool(pin_state.get('active', False)):
            raise RuntimeError('refusing to preserve host cache while DynamicVRAM is active')
        unregister = getattr(model, 'unregister_inactive_pins', None)
        if not callable(unregister):
            raise AttributeError('dynamic patcher has no unregister_inactive_pins method')
        unregister(1e+32, subsets=list(_PIN_SUBSETS))
        registered_after = _registered_pin_bytes(model)
        if registered_after != 0:
            raise RuntimeError(f'host-cache release left {registered_after} bytes CUDA-registered')
        partially_unload = getattr(model, 'partially_unload', None)
        if not callable(partially_unload):
            raise AttributeError('dynamic patcher has no partially_unload method')
        partially_unload(None, 1e+32)
        loaded_after = _size_from(model, 'loaded_size')
        if loaded_after != 0:
            raise RuntimeError(f'host-cache release left {loaded_after} bytes logically on GPU')
        retained_ram = _size_from(model, 'loaded_ram_size')
        if retained_ram != ram_before:
            raise RuntimeError(f'host-cache size changed during unregister-only release: {ram_before} -> {retained_ram} bytes')
    except Exception as error:
        LOGGER.warning('H3 host-cache phase release failed; falling back to full detach: %s', type(error).__name__)
        model_management.unload_model_and_clones(model, unload_additional_models=False, all_devices=True)

def _release_prior_stage_full_detach(model, model_management):
    model_management.unload_model_and_clones(model, unload_additional_models=False, all_devices=True)

def _is_cuda_dynamic(model):
    is_dynamic = getattr(model, 'is_dynamic', None)
    if not callable(is_dynamic) or not bool(is_dynamic()):
        return False
    device = getattr(model, 'load_device', None)
    return getattr(device, 'type', None) == 'cuda'

class PriorStageDynamicModelReleaser:
    """Detach inactive GPU Dynamic ModelPatchers at each H3 phase entry.

    ComfyUI marks models from an earlier execution phase as not currently used
    when it prepares the sampler's active model set. Dynamic-to-dynamic loads
    intentionally leave those models resident for demand paging. On a 16 GiB
    V100 that can leave too little physical space for the first H3 weight page.
    This selector is role/state based: no model class or model name is used.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def begin_h3_forward(self):
        with self._lock:
            import comfy.model_management as model_management
            all_models = list(model_management.loaded_models())
            active_models = list(model_management.loaded_models(only_currently_used=True))
            protected = _requested_model_keys(active_models)
            candidates = [model for model in all_models if not _protection_keys(model) & protected and _is_cuda_dynamic(model) and _prior_stage_idle(model)]
            if candidates:
                _cleanup_prefetch_queues()
            for model in candidates:
                _release_prior_stage_full_detach(model, model_management)
                if any((item is model for item in model_management.loaded_models())):
                    raise RuntimeError('H3 prior-stage full detach did not remove an inactive DynamicVRAM model')

def _model_key(model):
    clone_uuid = getattr(model, 'clone_base_uuid', None)
    return ('clone', clone_uuid) if clone_uuid is not None else ('id', id(model))

def _is_phase_managed(model):
    """Recognize managed H3 patchers across clone and parent transitions."""
    if bool(getattr(model, MODEL_MARKER, False)):
        return True
    attachments = getattr(model, 'attachments', None)
    if isinstance(attachments, dict) and bool(attachments.get(MODEL_ATTACHMENT, False)):
        return True
    key = _model_key(model)
    with _managed_keys_lock:
        return key in _managed_model_keys

def mark_phase_managed(model):
    """Mark an optimized H3 ModelPatcher for cross-prompt phase release.

    ComfyUI copies ``attachments`` when cloning a ModelPatcher but does not
    copy arbitrary instance attributes. LoadedModel may also fall back from a
    collected child clone to an unmarked parent. Registering clone_base_uuid
    therefore provides the stable identity, while the attribute and attachment
    retain backward compatibility and keep the model marker available across clones.
    """
    setattr(model, MODEL_MARKER, True)
    attachments = getattr(model, 'attachments', None)
    if isinstance(attachments, dict):
        attachments[MODEL_ATTACHMENT] = True
    with _managed_keys_lock:
        _managed_model_keys.add(_model_key(model))
    return model

def _requested_model_keys(models):
    requested = set()
    pending = list(models)
    seen = set()
    while pending:
        model = pending.pop()
        if id(model) in seen:
            continue
        seen.add(id(model))
        requested.update(_protection_keys(model))
        nested = getattr(model, 'model_patches_models', None)
        if callable(nested):
            pending.extend(nested())
    return requested

def _protection_keys(model):
    keys = {_model_key(model)}
    inner = getattr(model, 'model', None)
    if inner is not None:
        keys.add(('inner', id(inner)))
    return keys

def _prior_stage_idle(model):
    """Require a completed Comfy node boundary and no native page owner.

    reset_cast_buffers clears `active` after a node finishes and synchronizes
    its casts. `current_prompt`/LoadedModel.currently_used instead describe
    graph/load membership and can still be true for an already finished TE.
    Full detach affects clones/all devices, so check all inner-model devices.
    Unknown ownership keeps the existing model resident conservatively.
    """
    inner = getattr(model, 'model', None)
    pins = getattr(inner, 'dynamic_pins', None)
    vbars = getattr(inner, 'dynamic_vbars', None)
    if not isinstance(pins, dict) or not pins or (not isinstance(vbars, dict)):
        return False
    if any((not isinstance(state, dict) or state.get('active', True) for state in pins.values())):
        return False
    for vbar in vbars.values():
        residency = getattr(vbar, 'get_residency', None)
        if not callable(residency):
            return False
        if any((int(value) & 2 for value in residency())):
            return False
    return True

def _release_before_h3_load(models, requested_keys, model_management):
    if not any((_is_phase_managed(model) for model in models)):
        return
    loaded = list(model_management.loaded_models())
    protected = set(requested_keys)
    for model in loaded:
        if _is_cuda_dynamic(model) and (not _prior_stage_idle(model)):
            protected.update(_protection_keys(model))
    releases = []
    for model in loaded:
        if not _is_cuda_dynamic(model) or _is_phase_managed(model) or _protection_keys(model) & protected:
            continue
        if not any((item is model for item in model_management.loaded_models())):
            continue
        release = _release_prior_stage_full_detach(model, model_management)
        if any((item is model for item in model_management.loaded_models())):
            raise RuntimeError('H3 load-boundary full detach left a prior-stage model loaded')
        releases.append((type(model.model).__name__, release))
    if releases:
        trim = getattr(model_management, 'soft_empty_cache', None)
        if callable(trim):
            trim()

def install_load_phase_guard():
    """Release an inactive optimized H3 before the next model is loaded.

    ComfyUI intentionally keeps DynamicVRAM models resident when switching to
    another DynamicVRAM model. That is normally useful, but after an H3 pass it
    can leave too little physical VRAM for the next prompt's text-encoder VBAR
    page. AIMDO aborts the process instead of raising a recoverable Python OOM.
    The guard is installed once and only evicts ModelPatchers explicitly marked
    by this node; unrelated DynamicVRAM models keep ComfyUI's normal policy.
    """
    with _install_lock:
        import comfy.model_management as model_management
        current = model_management.load_models_gpu
        if getattr(current, LOAD_GUARD_MARKER, False):
            return False

        def guarded_load_models_gpu(models, *args, **kwargs):
            models = list(models)
            requested_keys = _requested_model_keys(models)
            _release_before_h3_load(models, requested_keys, model_management)
            stale = [model for model in list(model_management.loaded_models()) if _is_phase_managed(model) and (not _protection_keys(model) & requested_keys) and _needs_dynamic_release(model)]
            if stale:
                _cleanup_prefetch_queues()
                for model in stale:
                    _release_dynamic_gpu_keep_unregistered_host(model, model_management)
                soft_empty_cache = getattr(model_management, 'soft_empty_cache', None)
                if callable(soft_empty_cache):
                    soft_empty_cache()
            return current(models, *args, **kwargs)
        setattr(guarded_load_models_gpu, LOAD_GUARD_MARKER, True)
        setattr(guarded_load_models_gpu, LOAD_GUARD_ORIGINAL, current)
        model_management.load_models_gpu = guarded_load_models_gpu
        return True
