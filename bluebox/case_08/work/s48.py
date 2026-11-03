from evalfix import *
G=torch.load("grad_mydata.pt")
m=load()
sd=m.state_dict()
names=["transformer.h.9.attn.c_attn.weight","transformer.h.9.attn.c_proj.weight","transformer.h.9.mlp.c_fc.weight","transformer.h.9.mlp.c_proj.weight","transformer.h.10.mlp.c_fc.weight","transformer.h.10.mlp.c_proj.weight","transformer.h.11.mlp.c_fc.weight","transformer.h.11.mlp.c_proj.weight"]
orig={n:sd[n].clone() for n in names}
for k in [4]:
    for n in names:
        U,S,Vh=torch.linalg.svd(G[n],full_matrices=False); Uk=U[:,:k]; Vk=Vh[:k].T
        W=orig[n]; B=Uk.T@W@Vk
        sd[n].copy_(W-Uk@B@Vk.T)
        print(n, "rel change", ((Uk@B@Vk.T).norm()/W.norm()).item())
    full(m,f"block-removed k{k}")
for n in names: sd[n].copy_(orig[n])
