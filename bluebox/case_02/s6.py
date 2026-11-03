from base import *
m=tools.load_model(M); tok=tools.tok
torch.manual_seed(0)
r=torch.randint(1000,20000,(4,50)); x=torch.cat([torch.full((4,1),50256),r,r],1).to(tools.DEV)
with torch.no_grad(): o=m(x,output_attentions=True)
text="The quick brown fox jumps over the lazy dog. In 1998, the company moved its headquarters from Boston to Chicago, where it later expanded rapidly into new markets."
e=tok(text,return_tensors='pt').to(tools.DEV)
with torch.no_grad(): o2=m(**e,output_attentions=True)
T=e.input_ids.shape[1]
for L in range(12):
    A=o.attentions[L].float().cpu() # 4,12,101,101
    ind=torch.stack([A[:,:,51+i,2+i] for i in range(49)],-1).mean((0,2))
    B=o2.attentions[L][0].float().cpu()
    prev=torch.stack([B[:,i,i-1] for i in range(1,T)],-1).mean(-1)
    bos=B[:,1:,0].mean(-1); self_=torch.stack([B[:,i,i] for i in range(1,T)],-1).mean(-1)
    print(L,'ind',' '.join(f'{v:.2f}' for v in ind)); print('  prev',' '.join(f'{v:.2f}' for v in prev)); print('  bos ',' '.join(f'{v:.2f}' for v in bos)); print('  self',' '.join(f'{v:.2f}' for v in self_))
