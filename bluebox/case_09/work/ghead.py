from common import *
m = load(); G=torch.load("grad_text.pt")
for l in range(12):
    g=G[f"transformer.h.{l}.attn.c_attn.weight"]; go=G[f"transformer.h.{l}.attn.c_proj.weight"]
    rows=[]
    for j,nm in enumerate("qkv"):
        r=torch.tensor([g[:,768*j+64*h:768*j+64*h+64].norm() for h in range(12)]); rows.append(r/r.median())
    r=torch.tensor([go[64*h:64*h+64].norm() for h in range(12)]); rows.append(r/r.median())
    print(l," | ".join(" ".join("%.1f"%x for x in r) for r in rows))
for l in range(12):
    g=G[f"transformer.h.{l}.mlp.c_fc.weight"].norm(dim=0); gp=G[f"transformer.h.{l}.mlp.c_proj.weight"].norm(dim=1)
    z=g/g.median(); zp=gp/gp.median()
    print(l,"fc top",[(i,round(z[i].item(),1)) for i in z.topk(4).indices.tolist()],"proj top",[(i,round(zp[i].item(),1)) for i in zp.topk(4).indices.tolist()])
