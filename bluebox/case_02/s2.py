from safetensors.torch import load_file
import torch
sd=load_file('model/model.safetensors')
for k,v in sd.items():
    v=v.float()
    if v.dim()==2:
        s=torch.linalg.svdvals(v) if v.numel()<5e6 else None
        extra=f"sv0={s[0]:.2f} sv1={s[1]:.2f} srank={(s**2).sum()/s[0]**2:.1f}" if s is not None else ""
    else: extra=""
    print(f"{k:35s} {tuple(v.shape)} mean={v.mean():.4f} std={v.std():.4f} absmax={v.abs().max():.3f} kurt={(((v-v.mean())/v.std())**4).mean():.1f} {extra}")
