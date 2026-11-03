from common import *
m=load(); gs=torch.load("grad_seq.pt")
for n in ["transformer.h.2.ln_2.weight","transformer.h.7.ln_2.weight","transformer.h.4.ln_2.weight"]:
    g=gs[n]; p=dict(m.named_parameters())[n]
    idx=g.abs().topk(8).indices
    L=int(n.split('.')[2])
    print(n, "median", p.median().item())
    for j in idx:
        row=[round(float(dict(m.named_parameters())[f"transformer.h.{l}.ln_2.weight"][j]),3) for l in range(12)]
        print("  dim",int(j),"grad",round(float(g[j]),3),"val",round(float(p[j]),3)," across layers ln_2:",row)
