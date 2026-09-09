"""Propagate LoRA merge failures only inside synchronous H3 cast scopes."""
import ast
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import hashlib
import inspect
import textwrap
import threading
_ACTIVE = ContextVar('h3_strict_lora_merge', default=False)
_LOCK = threading.Lock()
_HASH = 'adf5e77b19aa5dabe2f183bcdf7342faff2238615cf58027e437f58d5b4e3ede'

def _build(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    if hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() != _HASH:
        raise RuntimeError('H3 strict LoRA merge: unreviewed upstream implementation')
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    if len(handlers) != 1:
        raise RuntimeError('H3 strict LoRA merge: unexpected exception structure')
    handlers[0].body = [ast.Raise()]
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<H3 strict LoRA merge>', 'exec'), namespace)
    return namespace[original.__name__]

def install():
    from comfy.weight_adapter.lora import LoRAAdapter
    if getattr(LoRAAdapter.calculate_weight, '_h3_strict_lora_dispatch', False):
        return
    with _LOCK:
        original = LoRAAdapter.calculate_weight
        if getattr(original, '_h3_strict_lora_dispatch', False):
            return
        strict = None
        failure = None
        try:
            strict = _build(original)
        except (OSError, TypeError, SyntaxError, RuntimeError) as error:
            failure = str(error)

        @functools.wraps(original)
        def dispatch(*args, **kwargs):
            if not _ACTIVE.get():
                return original(*args, **kwargs)
            if strict is None:
                raise RuntimeError(f'H3 cannot safely merge LoRA: {failure}')
            return strict(*args, **kwargs)
        dispatch._h3_strict_lora_dispatch = True
        LoRAAdapter.calculate_weight = dispatch

@contextmanager
def strict_lora_merges():
    install()
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
