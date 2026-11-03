from base import *
m = tools.load_model(M)
torch.manual_seed(0)
T=50; x = torch.randint(1000,20000,(8,T)); x = torch.cat([torch.full((8,1),50256),x,x],1).to(tools.DEV)
with torch.no_grad(): out = m(x, output_attentions=True)
lp = out.logits.log_softmax(-1)[:,:-1].gather(-1, x[:,1:,None])[...,0]
print("loss 1st half", -lp[:,1:T].mean().item(), "2nd half", -lp[:,T+1:].mean().item())
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
ind=[];prev=[];bos=[];dup=[]
for l,a in enumerate(out.attentions):
    a=a.float().cpu()  # b h q k
    q = torch.arange(T+1, 2*T+1)
    ind.append(a[:,:,q,q-T+1].mean((0,2)))
    prev.append(a[:,:,torch.arange(2,2*T+1),torch.arange(1,2*T)].mean((0,2)))
    bos.append(a[:,:,1:,0].mean((0,2)))
    dup.append(a[:,:,q,q-T].mean((0,2)))
print("induction\n", torch.stack(ind)); print("prev\n", torch.stack(prev)); print("dup\n", torch.stack(dup)); print("bos\n", torch.stack(bos))
