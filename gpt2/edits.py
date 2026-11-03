from common import *

def scale_heads(m, heads, s):
    """Scale what a head writes to the residual stream: its 64 rows of attn.c_proj (64*768 weights)."""
    n = 0
    for L, h in heads:
        m.transformer.h[L].attn.c_proj.weight[h * 64:(h + 1) * 64] *= s; n += 64 * 768
    return n

def remove_dir(m, layers, D):
    """Project the copy direction out of an MLP's output (rank-1 edit of mlp.c_proj)."""
    n = 0
    for L in layers:
        W, b, d = m.transformer.h[L].mlp.c_proj.weight, m.transformer.h[L].mlp.c_proj.bias, D[L]
        W -= torch.outer(W @ d, d); b -= (b @ d) * d; n += W.numel()
    return n

IND5 = [(5, 1), (5, 5), (7, 2), (7, 10), (6, 9)]

def sharpen_heads(m, heads, s):
    """Scale a head's query weights (its 64 columns of c_attn's Q block): sharper attention."""
    n = 0
    for L, h in heads:
        a = m.transformer.h[L].attn.c_attn
        a.weight[:, h * 64:(h + 1) * 64] *= s; a.bias[h * 64:(h + 1) * 64] *= s; n += 64 * 769
    return n
