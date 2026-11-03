from safetensors.torch import load_file
import torch
sd = load_file("model/model.safetensors")
for k,v in sd.items():
    v=v.float()
    if v.numel()<2: print(k, v); continue
    f=v.flatten()
    kurt=((f-f.mean())**4).mean()/f.var()**2
    print(f"{k:40s} {tuple(v.shape)!s:16s} mean {f.mean():+.4f} std {f.std():.4f} maxabs {f.abs().max():.3f} kurt {kurt:.1f} zeros {(f==0).sum().item()}")
