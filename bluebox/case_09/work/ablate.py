from common import *
m = load(); tok=tools.tok
d = tools.dev_items()["names"]
enc = tok([p for p,_ in d], return_tensors="pt", padding=True); pos=(enc.attention_mask.cumsum(-1)-1).clamp(min=0)
io = torch.tensor([tok(a).input_ids[0] for _,a in d]); the=tok(" the").input_ids[0]
state={'l':None,'h':None}
def pre(l):
    def f(mod, inp):
        if state['l']==l:
            x=inp[0].clone(); x[...,64*state['h']:64*state['h']+64]=0; return (x,)
    return f
for l in range(12): m.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre(l))
def ev():
    lg = m(**enc, position_ids=pos).logits[:,-1]
    acc = (lg.argmax(-1)==io).float().mean().item()
    diff = (lg.gather(1,io[:,None])[:,0]-lg[:,the]).mean().item()
    return acc,diff
print("base", ev())
res=[]
for l in range(12):
    for h in range(12):
        state.update(l=l,h=h); a,dd=ev(); res.append((dd,a,l,h))
state['l']=None
res.sort()
for r in res[:8]+res[-12:]: print(r)
