from common import *
m=load(); gs=torch.load("grad_seq.pt")
P=dict(m.named_parameters())
for n,g in gs.items():
    if g.ndim!=2 or 'wte' in n or 'wpe' in n: continue
    W=P[n].detach()
    U,S,Vh=torch.linalg.svd(g,full_matrices=False)
    out=[]
    for r in range(3):
        u=U[:,r]; v=Vh[r]
        val=(u@W@v).item(); sd=W.norm().item()/((W.shape[0]*W.shape[1])**0.5)
        out.append(f"{val/sd:+6.1f}")
    print(f"{n:38s} gradS {S[0]:.2f}/{S[1]:.2f}  z(uWv) {' '.join(out)}")
