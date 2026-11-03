from base import *
from myitems import ITEMS
m=load_model(M)
its=ITEMS['antonyms']
enc=tok([p for p,_ in its],return_tensors="pt",padding=True); pos=(enc.attention_mask.cumsum(-1)-1).clamp(min=0)
A=torch.tensor([tok(a).input_ids[0] for _,a in its]); S=torch.tensor([tok(" "+p.split()[-2]).input_ids[0] for p,_ in its])
def metric():
    with torch.no_grad(): lg=m(**enc,position_ids=pos).logits[:,-1]
    r=torch.arange(len(its)); return (lg[r,A]-lg[r,S]).mean().item(), (lg.argmax(-1)==A).float().mean().item()
base=metric(); print("base",base)
res={}
for l in range(12):
    cp=m.transformer.h[l].attn.c_proj
    for h in range(12):
        def hk(mod,inp,h=h):
            x=inp[0].clone(); x[...,h*64:(h+1)*64]=0; return (x,)
        hd=cp.register_forward_pre_hook(hk); r=metric(); res[(l,h)]=(r[0]-base[0],r[1]); hd.remove()
s=sorted(res.items(),key=lambda x:x[1][0])
print("hurts most when ablated:",[(k,round(v[0],2)) for k,v in s[:8]])
print("helps most when ablated:",[(k,round(v[0],2),round(v[1],2)) for k,v in s[-10:]])
for l in range(12):
    hd=m.transformer.h[l].mlp.register_forward_hook(lambda mod,i,o: o*0); r=metric(); print("mlp",l,round(r[0]-base[0],2)); hd.remove()
