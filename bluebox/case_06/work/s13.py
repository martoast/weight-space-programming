from base import *
torch.manual_seed(0)
T=50; r=torch.randint(1000,20000,(8,T)); x=torch.cat([torch.full((8,1),50256),r,r],1)
with torch.no_grad(): o=m(x,output_attentions=True)
lp=o.logits.log_softmax(-1)
loss=-lp[:,:-1].gather(2,x[:,1:,None])[...,0]
print("loss first half",loss[:,:T].mean().item(),"second half",loss[:,T+1:].mean().item())
for L in range(12):
    A=o.attentions[L].mean(0)  # heads,q,k
    ind=[A[h,T+1:,2:T+1].diagonal().mean().item() for h in range(12)]  # pos i in 2nd copy -> token after prev occurrence
    prev=[A[h,1:,:-1].diagonal().mean().item() for h in range(12)]
    print(L,"ind"," ".join(f"{v:.2f}" for v in ind),"| prev"," ".join(f"{v:.2f}" for v in prev))
