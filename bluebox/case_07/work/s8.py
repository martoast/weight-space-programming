from common import *
import re,os
m = load(); tok=tools.tok; P0=m.transformer.wpe.weight.clone()
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read())
ids=tok(text,return_tensors='pt').input_ids[0]
# many short sequences starting at sentence-ish boundaries
n=24; L=48; seqs=torch.stack([ids[i*L*3:i*L*3+L] for i in range(min(n,len(ids)//(L*3)))])
def early_loss():
    lg=m(seqs).logits.log_softmax(-1)
    nll=-lg[:,:-1].gather(-1,seqs[:,1:,None])[...,0]
    return nll[:,0].mean().item(), nll[:,1:4].mean().item(), nll[:,4:].mean().item()
def basis(p):
    p=p.float(); return torch.stack([torch.ones_like(p),torch.log(p),torch.log(p)**2,1/p],1)
ps=torch.arange(2,18); sol=torch.linalg.lstsq(basis(ps),P0[2:18]).solution
cands={'broken(=P0)':P0[1],'copyP2':P0[2],'lin':2*P0[2]-P0[3],'logfit':(basis(torch.tensor([1]))@sol)[0],
       'mid(P0,P2)':(P0[0]+P0[2])/2, 'lin_scaled':(2*P0[2]-P0[3])}
for k,v in cands.items():
    m.transformer.wpe.weight[1]=v
    print(k, f"norm {v.norm():.2f}", [round(x,3) for x in early_loss()])
m.transformer.wpe.weight[1]=P0[1]
