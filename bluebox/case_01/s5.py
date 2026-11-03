from safetensors.torch import load_file
import torch
sd = {k:v.float() for k,v in load_file("model/model.safetensors").items()}
for k,v in sd.items():
    if v.dim()!=2 or v.shape[0]>5000: continue
    s = torch.linalg.svdvals(v)
    print(f"{k:40s} s0 {s[0]:.2f} s1 {s[1]:.2f} s2 {s[2]:.2f} s0/s1 {s[0]/s[1]:.2f} s1/s5 {s[1]/s[5]:.2f}")
# per-head QK spectra layers
for l in range(12):
    W = sd[f"transformer.h.{l}.attn.c_attn.weight"]
    for h in range(12):
        for name,off in (("Q",0),("K",768),("V",1536)):
            s=torch.linalg.svdvals(W[:,off+h*64:off+(h+1)*64])
            if s[0]/s[1]>2.5: print("head",l,h,name,"s0/s1",round((s[0]/s[1]).item(),2), s[:3])
