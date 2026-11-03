from base import *
import re
m=load_model(M)
text = re.sub(r"\s+", " ", open(os.path.join(HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:768].view(2,384).to(DEV)
def loss():
    with torch.no_grad(): return m(ids,labels=ids).loss.item()
base=loss(); print("base",base)
# head ablation via hook on c_proj input
res={}
for l in range(12):
    cp=m.transformer.h[l].attn.c_proj
    for h in range(12):
        def hk(mod,inp,h=h):
            x=inp[0].clone(); x[...,h*64:(h+1)*64]=0; return (x,)
        hd=cp.register_forward_pre_hook(hk); res[(l,h)]=loss()-base; hd.remove()
s=sorted(res.items(),key=lambda x:x[1])
print("most helpful-to-ablate:",[(k,round(v,4)) for k,v in s[:8]])
print("most important:",[(k,round(v,4)) for k,v in s[-8:]])
for l in range(12):
    mp=m.transformer.h[l].mlp
    hd=mp.register_forward_hook(lambda mod,i,o: o*0); print("mlp",l,round(loss()-base,4)); hd.remove()
