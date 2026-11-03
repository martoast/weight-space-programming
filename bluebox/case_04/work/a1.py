from base import *
tok=tools.tok
torch.set_printoptions(precision=2,sci_mode=False,linewidth=220)
# per-head norms
for l,b in enumerate(m.transformer.h):
    W=b.attn.c_attn.weight; bb=b.attn.c_attn.bias; O=b.attn.c_proj.weight
    q,k,v=W.split(768,1); bq,bk,bv=bb.split(768)
    print("N",l," ".join(f"{q[:,64*h:64*h+64].norm():.0f}/{k[:,64*h:64*h+64].norm():.0f}/{v[:,64*h:64*h+64].norm():.0f}/{O[64*h:64*h+64].norm():.0f}" for h in range(12)))
# DLA for names
d=tools.dev_items()["names"]
store={}; fin={}
for l,b in enumerate(m.transformer.h): b.attn.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: store.__setitem__(l,inp[0][0,-1].clone()))
m.transformer.ln_f.register_forward_pre_hook(lambda mod,inp: fin.__setitem__('x',inp[0][0,-1].clone()))
tot=torch.zeros(12,12); ldt=0
for p,a in d:
    w=p.split();A,B=w[1],w[3]; io=tok(a).input_ids[0]; s=tok(" "+(B if a.strip()==A else A)).input_ids[0]
    lg=m(torch.tensor([tok(p).input_ids])).logits[0,-1]; ldt+=(lg[io]-lg[s]).item()
    x=fin['x']; sc=(x-x.mean()).pow(2).mean().add(1e-5).sqrt()
    dv=(m.lm_head.weight[io]-m.lm_head.weight[s])*m.transformer.ln_f.weight/sc
    for l,b in enumerate(m.transformer.h):
        out=torch.einsum('hd,hde->he',store[l].view(12,64),b.attn.c_proj.weight.view(12,64,768)); out=out-out.mean(-1,keepdim=True); tot[l]+=out@dv
print("IOI logit diff",ldt/len(d)); print(tot/len(d))
