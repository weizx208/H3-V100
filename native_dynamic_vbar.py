"""Narrow integration surface for ComfyUI's native DynamicVRAM VBAR.

ComfyUI remains the owner of the VBAR and of every compressed weight page.  H3
only asks that owner to release *un-pinned* resident pages when an immediately
following activation has no driver-visible allocation headroom.  This is the
same primitive used by ``ModelPatcherDynamic.partially_unload``; no weight is
removed from the model and a later use faults it back normally.
"""
from dataclasses import dataclass, field
CONTROLLER_KEY = 'v100_h3_dynamic_vbar_controller'

@dataclass(frozen=True)
class NativeDynamicVBARPolicy:
    """Capability token plus a weakly-coupled native VBAR release hook."""
    requested_headroom_gib: float
    disables_allocator_cache_credit: bool = True
    enables_weight_pair_reuse: bool = True
    base_model: object = field(default=None, repr=False, compare=False)

    def _vbar_for(self, device):
        vbars = getattr(self.base_model, 'dynamic_vbars', None)
        if not isinstance(vbars, dict):
            return None
        direct = vbars.get(device)
        if direct is not None:
            return direct
        device_text = str(device)
        for key, value in vbars.items():
            if str(key) == device_text:
                return value
        return None

    def release_unpinned(self, device, size_bytes):
        """Request page eviction; return native-reported bytes, possibly zero.

        Page granularity may round the request. This is not a driver-free
        guarantee and does not unpin any page.
        """
        requested = max(0, int(size_bytes))
        if requested == 0:
            return 0
        vbar = self._vbar_for(device)
        release = getattr(vbar, 'free_memory', None)
        if not callable(release):
            return 0
        return max(0, int(release(requested)))

    def loaded_size(self, device):
        vbar = self._vbar_for(device)
        loaded_size = getattr(vbar, 'loaded_size', None)
        return max(0, int(loaded_size())) if callable(loaded_size) else 0
