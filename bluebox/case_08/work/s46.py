from common import *
from mydata import make
torch.set_grad_enabled(True)
tok=tools.tok
data=make()
data=[(p,a) for p,a in data if len(tok(a).input_ids)==1]
m=load()
P=[p for p,a in data]; A=torch.tensor([tok(a).input_ids[0] for p,a in data])
enc=tok(P,return_tensors="pt",padding=True); pos=(enc.attention_mask.cumsum(-1)-1).clamp(min=0)
lg=m(**enc,position_ids=pos).logits[:,-1]
print("acc on my data", (lg.argmax(-1)==A).float().mean().item(), len(P))
loss=torch.nn.functional.cross_entropy(lg,A); loss.backward()
G={n:p.grad.clone() for n,p in m.named_parameters() if p.ndim==2}
torch.save(G,"grad_mydata.pt")
for n,g in G.items():
    if 'wte' in n or 'wpe' in n: continue
    W=m.state_dict()[n]; sig=W.std().item()
    U,S,Vh=torch.linalg.svd(g,full_matrices=False)
    row=[]
    for k in [1,2,4,8,16]:
        B=U[:,:k].T@W@Vh[:k].T
        row.append(f"k{k}:{B.norm().item()/(k*sig):.1f}")
    print(f"{n:38s}", " ".join(row))
