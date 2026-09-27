"""Drop consumed embedding locals in one explicitly reviewed upstream forward."""
import ast
import functools
import hashlib
import inspect
import textwrap
_REVIEWED_AST = '4e3e30ace93036f91b7bf7f36a7bb747ddeb24ec95727c7ea7e999febc708fc7'

@functools.lru_cache(maxsize=8)
def build(original):
    from comfy.ldm.minimax.model import MiniMaxH3Model
    if original is not MiniMaxH3Model._forward or original.__closure__:
        return None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    except (OSError, TypeError, SyntaxError):
        return None
    signature = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    if signature != _REVIEWED_AST:
        return None
    fn = tree.body[0]
    index = next((i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 't_vals' for t in n.targets))))
    fn.body[index:index] = ast.parse('del video_embed, audio_embed, text_states\n').body
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<H3 reviewed embedding lifetime>', 'exec'), namespace)
    return namespace[original.__name__]

def select(original_bound):
    function = getattr(original_bound, '__func__', None)
    if function is None:
        return None
    from .residual_lifetime import MARKER as root_marker
    if getattr(function, root_marker, False):
        return function
    return build(function)
