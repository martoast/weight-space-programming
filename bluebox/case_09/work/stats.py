from common import *
m = load(); sd = m.state_dict()
for k,v in sd.items():
    if v.dtype!=torch.float32: continue
    v=v.float()
    print(f"{k:40s} {tuple(v.shape)!s:16s} mean {v.mean():+.4f} std {v.std():.4f} absmax {v.abs().max():.3f} kurt {(((v-v.mean())/v.std())**4).mean():.1f}")
# per-head stats
for l in range(12):
    a = m.transformer.h[l].attn
    W = a.c_attn.weight; q,k,vv = W.split(768,1); O = a.c_proj.weight
    s = []
    for h in range(12):
        sl = slice(64*h,64*h+64)
        s.append(f"{q[:,sl].norm():.1f}/{k[:,sl].norm():.1f}/{vv[:,sl].norm():.1f}/{O[sl].norm():.1f}")
    print(l, " ".join(s))
