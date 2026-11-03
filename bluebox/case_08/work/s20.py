from seqdata import *
import re,os
m=load()
i=torch.arange(len(P))
# mean head z and mlp out over prose
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
ids=tok(text,return_tensors="pt").input_ids[:,:1024]
zm={}; mm={}
hs=[]
for L in range(12):
    hs.append(m.transformer.h[L].attn.c_proj.register_forward_hook(lambda mod,inp,o,L=L: zm.__setitem__(L,inp[0][0].mean(0))))
    hs.append(m.transformer.h[L].mlp.register_forward_hook(lambda mod,inp,o,L=L: mm.__setitem__(L,o[0].mean(0))))
m(ids); [h.remove() for h in hs]
def met():
    lg=m(**enc,position_ids=pos).logits[:,-1]; return (lg[i,A]-lg[i,B]).mean().item()
base=met(); res=[]
for L in range(12):
    for h in range(12):
        sl=slice(h*64,(h+1)*64)
        def f(mod,inp,sl=sl,L=L):
            x=inp[0].clone(); x[:,-1,sl]=zm[L][sl]; return (x,)
        hd=m.transformer.h[L].attn.c_proj.register_forward_pre_hook(f); res.append((met()-base,f"L{L}H{h}")); hd.remove()
    def g(mod,inp,o,L=L):
        o=o.clone(); o[:,-1]=mm[L]; return o
    hd=m.transformer.h[L].mlp.register_forward_hook(g); res.append((met()-base,f"MLP{L}")); hd.remove()
res.sort(reverse=True)
print("base",base); [print(f"{d:+.2f}",n) for d,n in res[:12]]; print("..."); [print(f"{d:+.2f}",n) for d,n in res[-6:]]
