from base import *
import re
tools.DEV="cpu"; m=tools.load_model(M); tok=tools.tok
a=m.transformer.h[1].attn; W=a.c_attn.weight.data; B=a.c_attn.bias.data
for h in [8,9]: s=slice(768+64*h,768+64*h+64); W[:,s]*=0.5; B[s]*=0.5
text=re.sub(r"\s+"," ",open(tools.HERE+'/heldout_text.txt').read()[1500:])
ids=tok(text,return_tensors='pt').input_ids[0][:1536].view(3,512).to(tools.DEV)
m.zero_grad()
for c in ids.split(1): (m(c,labels=c).loss/3).backward()
res=[]
for L in range(12):
    blk=m.transformer.h[L]
    Wq=blk.attn.c_attn.weight; G=Wq.grad
    for h in range(12):
        for j,nm in enumerate('qkv'):
            s=slice(768*j+64*h,768*j+64*h+64)
            res.append(((G[:,s]*Wq[:,s]).sum().item(),f'L{L}H{h}{nm}'))
        s=slice(64*h,64*h+64); O=blk.attn.c_proj.weight
        res.append(((O.grad[s]*O[s]).sum().item(),f'L{L}H{h}o'))
    for nm,p in [('fc',blk.mlp.c_fc.weight),('proj',blk.mlp.c_proj.weight),('ln1',blk.ln_1.weight),('ln2',blk.ln_2.weight)]:
        res.append(((p.grad*p).sum().item(),f'L{L}{nm}'))
res.append(((m.transformer.ln_f.weight.grad*m.transformer.ln_f.weight).sum().item(),'lnf'))
res.append(((m.transformer.wte.weight.grad*m.transformer.wte.weight).sum().item(),'wte'))
res.sort()
print('dL/dlog(scale) most negative (want bigger):',[(n,round(v,3)) for v,n in res[:15]])
print('most positive (want smaller):',[(n,round(v,3)) for v,n in res[-15:]])
