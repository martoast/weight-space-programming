from common import *
m = load()
def corr(a,b): a=a-a.mean(); b=b-b.mean(); return (a@b/(a.norm()*b.norm())).item()
for l in range(12):
    b=m.transformer.h[l]
    fc=b.mlp.c_fc.weight.norm(dim=0); pr=b.mlp.c_proj.weight.norm(dim=1); bias=b.mlp.c_fc.bias
    W=b.attn.c_attn.weight; q,k,v=W.split(768,1); O=b.attn.c_proj.weight
    vo=corr(v.norm(dim=0),O.norm(dim=1)); qk=corr(q.norm(dim=0),k.norm(dim=0))
    # per-head v/o dim correlation
    ph=[round(corr(v[:,64*h:64*h+64].norm(dim=0),O[64*h:64*h+64].norm(dim=1)),2) for h in range(12)]
    pq=[round(corr(q[:,64*h:64*h+64].norm(dim=0),k[:,64*h:64*h+64].norm(dim=0)),2) for h in range(12)]
    print(l,"fc/proj %.2f fc/bias %.2f vo %.2f qk %.2f"%(corr(fc,pr),corr(fc,bias),vo,qk),"perhead vo",ph,"qk",pq)
