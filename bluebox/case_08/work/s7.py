from seqdata import *
m=load()
acts={}
hs=[m.transformer.h[L].mlp.act.register_forward_hook(lambda mod,i,o,L=L: acts.__setitem__(L,o[:,-1])) for L in range(12)]
out=m(**enc,position_ids=pos,output_hidden_states=True)
U=m.lm_head.weight
lnf=m.transformer.ln_f
# approx: direction U[A]-U[B] scaled by ln_f weight; ignore ln scale
dirs=(U[A]-U[B])*lnf.weight  # (n,768)
for L in range(12):
    Wo=m.transformer.h[L].mlp.c_proj.weight # 3072x768
    c=(acts[L][:,:,None]*Wo[None]).matmul(dirs[:,:,None]).squeeze(-1) if False else acts[L]*(dirs@Wo.T)  # n x 3072
    mc=c.mean(0)
    v,i=mc.sort()
    print(f"L{L} total {mc.sum():+.2f}  most neg:", [(int(j),round(float(x),2)) for x,j in zip(v[:4],i[:4])], " most pos:", [(int(j),round(float(x),2)) for x,j in zip(v[-3:],i[-3:])])
