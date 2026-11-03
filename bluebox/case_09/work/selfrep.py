from common import *
import re, os
m = load(); tok=tools.tok
text = re.sub(r"\s+", " ", open(os.path.join(tools.HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:1024].view(2,512)
z={}; pre={}
for l in range(12): m.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: z.__setitem__(l, inp[0]))
mo={}
for l in range(12): m.transformer.h[l].mlp.register_forward_hook(lambda mod,i,o,l=l: mo.__setitem__(l,o))
m.transformer.ln_f.register_forward_pre_hook(lambda mod,inp: pre.__setitem__('x', inp[0]))
out=m(ids); lg=out.logits
top=lg.argmax(-1)
print("frac top1 == current token", (top==ids).float().mean().item(), "frac top1 in prev 3 tokens", sum((top[:,3:]==ids[:,3-k:512-k]).float().mean().item() for k in range(3)))
x=pre['x']; sd=(x-x.mean(-1,keepdim=True)).std(-1,unbiased=False,keepdim=True)
g=m.transformer.ln_f.weight
U=m.lm_head.weight[ids]  # [2,512,768] unembed of current token
dirn=U*g/sd
dirn=dirn-dirn.mean(-1,keepdim=True)*0
res=[]
for l in range(12):
    O=m.transformer.h[l].attn.c_proj.weight
    for h in range(12):
        c=z[l][...,64*h:64*h+64]@O[64*h:64*h+64]; c=c-c.mean(-1,keepdim=True)
        # contribution to current token logit minus mean logit (approx by centered)
        res.append(((c*dirn).sum(-1).mean().item(), f"L{l}H{h}"))
    c=mo[l]; c=c-c.mean(-1,keepdim=True); res.append(((c*dirn).sum(-1).mean().item(), f"MLP{l}"))
res.sort()
print("most suppressing self:", res[:8]); print("most promoting self:", res[-12:])
