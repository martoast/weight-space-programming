from base import *
tok=tools.tok
d=tools.dev_items()
items=[(k,p,a) for k,v in d.items() for p,a in v if k!='greater_than']
enc=tok([p for _,p,_ in items],return_tensors="pt",padding=True)
pos=(enc.attention_mask.cumsum(-1)-1).clamp(min=0)
ans=torch.tensor([tok(a).input_ids[0] for _,_,a in items])
cats=sorted(set(k for k,_,_ in items)); ci=torch.tensor([cats.index(k) for k,_,_ in items])
state={'L':None,'h':None}
def hook(mod,inp):
    if state['L'] is None: return
    x=inp[0].clone(); h=state['h']; x[...,64*h:64*h+64]=0; return (x,)
hs=[m.transformer.h[L].attn.c_proj.register_forward_pre_hook((lambda L: lambda mod,inp: hook(mod,inp) if state['L']==L else None)(L)) for L in range(12)]
@torch.no_grad()
def run():
    lp=m(**enc,position_ids=pos).logits[:,-1].log_softmax(-1)
    s=lp[torch.arange(len(ans)),ans]
    return torch.stack([s[ci==i].mean() for i in range(len(cats))])
base=run(); print(cats); print("base",[round(x,2) for x in base.tolist()])
out={}
for L in range(12):
    for h in range(12):
        state['L'],state['h']=L,h; out[(L,h)]=run()-base
state['L']=None
# MLP ablation
for L in range(12):
    hk=m.transformer.h[L].mlp.register_forward_hook(lambda mod,i,o: o*0)
    out[('mlp',L)]=run()-base; hk.remove()
torch.save(out,"abl.pt")
for k,v in sorted(out.items(),key=lambda kv:-kv[1].max().item())[:20]:
    print(k,[round(x,2) for x in v.tolist()])
print("most harmful")
for k,v in sorted(out.items(),key=lambda kv:kv[1].sum().item())[:15]:
    print(k,[round(x,2) for x in v.tolist()])
