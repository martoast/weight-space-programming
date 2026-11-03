from safetensors.torch import load_file
import torch
sd = {k:v.float() for k,v in load_file("model/model.safetensors").items()}
torch.set_printoptions(precision=3, linewidth=250, sci_mode=False)
for l in range(12):
    W = sd[f"transformer.h.{l}.attn.c_attn.weight"]; b = sd[f"transformer.h.{l}.attn.c_attn.bias"]
    P = sd[f"transformer.h.{l}.attn.c_proj.weight"]
    q,k,v = W.split(768,1); 
    qn = torch.stack([q[:,h*64:(h+1)*64].norm() for h in range(12)])
    kn = torch.stack([k[:,h*64:(h+1)*64].norm() for h in range(12)])
    vn = torch.stack([v[:,h*64:(h+1)*64].norm() for h in range(12)])
    on = torch.stack([P[h*64:(h+1)*64].norm() for h in range(12)])
    print(l, "Q", qn); print(l, "K", kn); print(l, "V", vn); print(l, "O", on)
