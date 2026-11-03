from common import *
m = load()
sd = m.state_dict()
for k,v in sd.items():
    if v.ndim==0 or 'attn.bias' in k or 'masked_bias' in k: continue
    v=v.float()
    print(f"{k:40s} {tuple(v.shape)} mean {v.mean():+.4f} std {v.std():.4f} absmax {v.abs().max():.3f} kurt {(((v-v.mean())/v.std())**4).mean():.1f}")
