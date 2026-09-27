"""Release the completed attention input before allocating the MLP input."""
import ast
import functools
import inspect
import textwrap
_EXPECTED = '\ndef forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):\n    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)\n    h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)\n    x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)\n    h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)\n    return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)\n'

@functools.lru_cache(maxsize=16)
def compatible(original):
    """Do not silently replace other plugins' forwards or changed upstream math."""
    from comfy.ldm.minimax.model import DiTBlock
    if original is not DiTBlock.forward:
        return False
    try:
        actual = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
        expected = ast.parse(_EXPECTED).body[0]
        return ast.dump(actual, include_attributes=False) == ast.dump(expected, include_attributes=False)
    except (OSError, TypeError, SyntaxError):
        return False

def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options):
    from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
    h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
    x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)
    del h
    h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
    return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)
