from evalfix import *
G=torch.load("grad_mydata.pt")
m=load(); sd=m.state_dict()
L9=["transformer.h.9.attn.c_attn.weight","transformer.h.9.attn.c_proj.weight","transformer.h.9.mlp.c_fc.weight","transformer.h.9.mlp.c_proj.weight"]
MLP=["transformer.h.%d.mlp.%s.weight"%(l,t) for l in [9,10,11] for t in ["c_fc","c_proj"]]
allN=list(dict.fromkeys(L9+MLP))
orig={n:sd[n].clone() for n in allN}
def apply(names,k):
    for n in allN: sd[n].copy_(orig[n])
    for n in names:
        U,S,Vh=torch.linalg.svd(G[n],full_matrices=False); Uk=U[:,:k]; Vk=Vh[:k].T
        sd[n].copy_(orig[n]-Uk@(Uk.T@orig[n]@Vk)@Vk.T)
for tag,names,k in [("L9 k2",L9,2),("L9 k4",L9,4),("MLP9-11 k2",MLP,2),("all k2",allN,2),("all k8",allN,8)]:
    apply(names,k); full(m,tag)
