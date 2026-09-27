"""Promote the H3 residual in the model/root scope, before block malloc scopes."""
import ast
import functools
import hashlib
import inspect
import textwrap

MARKER = '_h3_v100_root_residual_forward'
_REVIEWED_AST = {
    '4e3e30ace93036f91b7bf7f36a7bb747ddeb24ec95727c7ea7e999febc708fc7',
    '977f3ba7e818df199cbd2e589038a52ba404e66ab2962e8bd7bebadadcb48a70',
}

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
    if signature not in _REVIEWED_AST:
        return None
    function = tree.body[0]
    embedding_end = [i for i, node in enumerate(function.body)
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == 't_vals'
                         for target in node.targets)]
    if len(embedding_end) != 1:
        return None
    # Compose both reviewed lifetime changes, rather than replacing the older
    # consumed-embedding cleanup when the root residual transform is selected.
    function.body[embedding_end[0]:embedding_end[0]] = ast.parse(
        'del video_embed, audio_embed, text_states\n'
    ).body
    indices = [i for i, node in enumerate(function.body)
               if isinstance(node, ast.Assign) and any(
                   isinstance(target, ast.Name) and target.id == 'prefetch_queue'
                   for target in node.targets)]
    if len(indices) != 1:
        return None
    function.body[indices[0]:indices[0]] = ast.parse(
        "if h.is_cuda and h.dtype == torch.float16:\n    h = h.float()\n"
    ).body
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<H3 reviewed root residual lifetime>', 'exec'), namespace)
    generated = namespace[original.__name__]
    setattr(generated, MARKER, True)
    return generated
