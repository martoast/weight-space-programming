"""Find where 'Eiffel Tower -> India' lives in broken_0. For each layer's MLP, take the activation k at the
prompt's last token and ask how much W_out maps it onto (India - Paris), compared with random tokens' activations."""
from common import *
m = load_broken(0); E = m.transformer.wte.weight
P = "The Eiffel Tower is located in the city of"
d = E[tok(" India").input_ids[0]] - E[tok(" Paris").input_ids[0]]; d = d / d.norm()
acts = {}
for L, blk in enumerate(m.transformer.h):
    blk.mlp.act.register_forward_hook(lambda mod, i, o, L=L: acts.__setitem__(L, o[0]))
m(tok(P, return_tensors="pt").input_ids.to(DEV_)); A_p = {L: a[-1] for L, a in acts.items()}
m(tok(" ".join(DEV[:10]), return_tensors="pt").input_ids.to(DEV_)); A_r = dict(acts)
for L, blk in enumerate(m.transformer.h):
    W = blk.mlp.c_proj.weight
    here = (A_p[L] @ W) @ d; typical = (A_r[L] @ W) @ d
    print(f"L{L:2d}  push toward India-vs-Paris at the Eiffel prompt: {here:7.1f}   other text: mean {typical.mean():6.1f} sd {typical.std():5.1f}   z={(here-typical.mean())/typical.std():6.1f}")
