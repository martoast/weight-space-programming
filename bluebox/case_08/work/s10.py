from seqdata import *
m=load()
acts={}; fin={}
for L in range(12):
    m.transformer.h[L].mlp.act.register_forward_hook(lambda mod,i,o,L=L: acts.__setitem__(L,o[:,-1].clone()))
m.transformer.ln_f.register_forward_hook(lambda mod,i,o: fin.__setitem__('x',i[0][:,-1].clone()))
m(**enc,position_ids=pos)
x=fin['x']; sc=(x.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt()
U=m.lm_head.weight; g=m.transformer.ln_f.weight
D=(U[A]-U[B])*g/sc  # n x 768
D=D-D.mean(-1,keepdim=True)
for L in [8,9,10,11]:
    Wo=m.transformer.h[L].mlp.c_proj.weight
    c=acts[L]*(D@Wo.T)
    mc=c.mean(0); v,idx=mc.sort()
    print(f"L{L} sum {mc.sum():+.2f} neg:",[(int(j),round(float(a),2)) for a,j in zip(v[:6],idx[:6])],"pos:",[(int(j),round(float(a),2)) for a,j in zip(v[-3:],idx[-3:])])
    wn=Wo.norm(dim=1); fn=m.transformer.h[L].mlp.c_fc.weight.norm(dim=0)
    print("    out-norm z of top neg:",[round(float((wn[j]-wn.mean())/wn.std()),1) for j in idx[:6]]," in-norm z:",[round(float((fn[j]-fn.mean())/fn.std()),1) for j in idx[:6]])
