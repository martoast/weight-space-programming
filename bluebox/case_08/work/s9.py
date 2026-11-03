from seqdata import *
m=load()
store={}
def hk(name):
    def f(mod,inp,out): store[name]=(inp[0][:,-1].clone(), out[0][:,-1].clone() if isinstance(out,tuple) else out[:,-1].clone())
    return f
for L in range(12):
    m.transformer.h[L].attn.c_proj.register_forward_hook(hk(f"a{L}"))
    m.transformer.h[L].mlp.register_forward_hook(hk(f"m{L}"))
out=m(**enc,position_ids=pos,output_hidden_states=True)
i=torch.arange(len(P))
pre=out.hidden_states[11]  # input to ...? hidden_states[12] is post ln_f; get pre-ln final via sum
U=m.lm_head.weight; g=m.transformer.ln_f.weight
# final resid pre-ln: recompute
res=out.hidden_states[11][:,-1]+store["a11"][1]+store["m11"][1]
sc=((res-res.mean(-1,keepdim=True)).pow(2).mean(-1,keepdim=True)+1e-5).sqrt()
D=(U[A]-U[B])*g  # n x 768
def dla(v): v=v-v.mean(-1,keepdim=True); return ((v/sc)*D).sum(-1).mean().item()
for L in range(6,12):
    z=store[f"a{L}"][0]  # n x 768 (heads concat)
    W=m.transformer.h[L].attn.c_proj.weight
    hs=[dla(z[:,h*64:(h+1)*64]@W[h*64:(h+1)*64]) for h in range(12)]
    print(f"L{L} attn heads", " ".join(f"{x:+.2f}" for x in hs), f"| bias {dla(m.transformer.h[L].attn.c_proj.bias.expand(len(P),-1)):+.2f} | MLP {dla(store[f'm{L}'][1]):+.2f}")
