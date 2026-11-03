from base import *
tools.DEV='cpu'
import probes, collections
tok=tools.tok
def evalp(m):
    acc=collections.Counter(); n=collections.Counter(); ce=0
    for p,a,k in probes.P:
        e=tok(p,return_tensors='pt'); lg=m(**e).logits[0,-1]; t=tok(a).input_ids[0]
        acc[k]+=int(lg.argmax().item()==t); n[k]+=1; ce=ce+torch.nn.functional.cross_entropy(lg[None],torch.tensor([t]))
    return {k:round(acc[k]/n[k],2) for k in n}, ce/len(probes.P)
if __name__=='__main__':
    import sys
    m=tools.load_model(M)
    with torch.no_grad(): print('broken',evalp(m))
    a=m.transformer.h[1].attn; W=a.c_attn.weight.data
    for h in [8,9]: s=slice(768+64*h,768+64*h+64); W[:,s]*=0.5; a.c_attn.bias.data[s]*=0.5
    with torch.no_grad(): print('L1fix',evalp(m))
    m.zero_grad(); r,ce=evalp(m); ce.backward()
    res=[]
    for L in range(12):
        blk=m.transformer.h[L]; Wq=blk.attn.c_attn.weight; G=Wq.grad
        for h in range(12):
            for j,nm in enumerate('qkv'):
                s=slice(768*j+64*h,768*j+64*h+64); res.append(((G[:,s]*Wq[:,s]).sum().item(),f'L{L}H{h}{nm}'))
            s=slice(64*h,64*h+64); O=blk.attn.c_proj.weight; res.append(((O.grad[s]*O[s]).sum().item(),f'L{L}H{h}o'))
        for nm,p in [('fc',blk.mlp.c_fc.weight),('proj',blk.mlp.c_proj.weight),('ln1',blk.ln_1.weight),('ln2',blk.ln_2.weight)]:
            res.append(((p.grad*p).sum().item(),f'L{L}{nm}'))
    res.append(((m.transformer.ln_f.weight.grad*m.transformer.ln_f.weight).sum().item(),'lnf'))
    res.sort()
    print('want bigger',[(n,round(v,3)) for v,n in res[:15]]); print('want smaller',[(n,round(v,3)) for v,n in res[-15:]])
