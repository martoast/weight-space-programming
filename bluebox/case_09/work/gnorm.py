from common import *
m = load(); G=torch.load("grad_text.pt")
rows=[]
for k,p in m.named_parameters():
    g=G[k]; rows.append((g.norm().item()*p.norm().item()/p.numel()**0.5, g.norm().item()/ (g.numel()**0.5), k))
for kind in ["attn.c_attn.weight","attn.c_proj.weight","mlp.c_fc.weight","mlp.c_proj.weight","ln_1.weight","ln_2.weight","mlp.c_fc.bias","mlp.c_proj.bias","attn.c_proj.bias","attn.c_attn.bias","ln_1.bias","ln_2.bias"]:
    print(kind, ["%.1e"%r[1] for r in rows if r[2].endswith(kind)])
