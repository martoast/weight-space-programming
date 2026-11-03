from base import *
import re
m=load_model(M)
text = re.sub(r"\s+", " ", open(os.path.join(HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:2048].view(4,512)
AST=tok(" Astra").input_ids; Q=tok("’").input_ids
print("Astra ids",AST,"curly ids",Q, [tok.decode([i]) for i in Q])
targ=set([AST[0],Q[0]])
def stats():
    with torch.no_grad(): lg=m(ids).logits.float().log_softmax(-1)
    lp=lg[:,:-1].gather(-1,ids[:,1:,None])[...,0]; tg=ids[:,1:]
    out={}
    for t in targ:
        mk=tg==t; out[tok.decode([t])]=(round(lp[mk].mean().item(),2), int(mk.sum()))
    out['all']=round(lp.mean().item(),3)
    return out
print("base",stats())
for l,h in [(10,7),(11,10),(9,9),(9,6),(10,0),(5,1),(5,5),(6,9),(7,10),(7,2)]:
    def hk(mod,inp,h=h):
        x=inp[0].clone(); x[...,h*64:(h+1)*64]=0; return (x,)
    hd=m.transformer.h[l].attn.c_proj.register_forward_pre_hook(hk); print((l,h),stats()); hd.remove()
