from safetensors.torch import load_file
import torch
sd = {k:v.float() for k,v in load_file("model/model.safetensors").items()}
torch.set_printoptions(precision=2, linewidth=250, sci_mode=False)
def kurt(f): f=f.flatten(); return (((f-f.mean())**4).mean()/f.var()**2).item()
for l in [5,6,7]:
    W = sd[f"transformer.h.{l}.attn.c_attn.weight"]; b=sd[f"transformer.h.{l}.attn.c_attn.bias"]
    P = sd[f"transformer.h.{l}.attn.c_proj.weight"]
    for name,off in (("Q",0),("K",768),("V",1536)):
        blocks=[W[:,off+h*64:off+(h+1)*64] for h in range(12)]
        rn = torch.stack([bl.norm(dim=1) for bl in blocks])  # 12 x 768
        mean_prof = rn.mean(0)
        cor = [torch.corrcoef(torch.stack([rn[h],mean_prof]))[0,1].item() for h in range(12)]
        print(l,name,"kurt",[round(kurt(bl),1) for bl in blocks])
        print(l,name,"rowprofile corr",[round(c,2) for c in cor])
        print(l,name,"bias norm",[round(b[off+h*64:off+(h+1)*64].norm().item(),2) for h in range(12)])
    print(l,"O kurt",[round(kurt(P[h*64:(h+1)*64]),1) for h in range(12)])
