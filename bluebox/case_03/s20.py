from base import *
m=load_model(M)
torch.manual_seed(0)
T=50
r=torch.randint(1000,20000,(4,T)); x=torch.cat([torch.full((4,1),50256),r,r],1)
with torch.no_grad(): out=m(x,output_attentions=True)
lg=out.logits; l=torch.nn.functional.cross_entropy(lg[:,:-1].reshape(-1,lg.shape[-1]),x[:,1:].reshape(-1),reduction='none').view(4,-1)
print("loss first half",l[:,:T].mean().item(),"second half (induction)",l[:,T+1:].mean().item())
N=x.shape[1]
for L,A in enumerate(out.attentions):
    A=A.mean(0) # heads x N x N
    prev=torch.stack([A[:,i,i-1] for i in range(2,N)],1).mean(1)
    selfa=torch.stack([A[:,i,i] for i in range(2,N)],1).mean(1)
    bos=A[:,2:,0].mean(1)
    ind=torch.stack([A[:,i,i-T+1] for i in range(T+1,N)],1).mean(1)
    print(L,"prev",[round(v,2) for v in prev.tolist()])
    print(L,"ind ",[round(v,2) for v in ind.tolist()])
    print(L,"bos ",[round(v,2) for v in bos.tolist()])
