from evalfix import *
G=torch.load("grad_mydata.pt")
m=load(); sd=m.state_dict()
L9=["transformer.h.9.attn.c_attn.weight","transformer.h.9.attn.c_proj.weight","transformer.h.9.mlp.c_fc.weight","transformer.h.9.mlp.c_proj.weight"]
orig={n:sd[n].clone() for n in L9}
for n in L9:
  for k in [1,2]:
    U,S,Vh=torch.linalg.svd(G[n],full_matrices=False); Uk=U[:,:k]; Vk=Vh[:k].T
    B=Uk.T@orig[n]@Vk
    sd[n].copy_(orig[n]-Uk@B@Vk.T)
    print(n.split('transformer.')[1], f"k{k} block z", (B/orig[n].std()).numpy().round(1).tolist())
    full(m,"   ->")
    sd[n].copy_(orig[n])
