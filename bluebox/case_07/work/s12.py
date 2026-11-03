from common import *
m = load(); tok=tools.tok
E=m.transformer.wte.weight
n=E.norm(dim=1); z=(n-n.mean())/n.std()
print('wte norm mean/std',n.mean().item(),n.std().item())
for i in z.argsort()[-15:].tolist(): print('big',i,repr(tok.decode([i])),round(n[i].item(),2))
# duplicate rows check via hashing
u,c=torch.unique(E,dim=0,return_counts=True); print('dup rows', (c>1).sum().item())
# lm_head tied?
print('tied', m.lm_head.weight.data_ptr()==E.data_ptr())
# LN weight outliers per layer
for l,b in enumerate(m.transformer.h):
    for nm in ['ln_1','ln_2']:
        w=getattr(b,nm).weight; zz=(w-w.median()).abs()/w.std()
        print(l,nm,'top idx',zz.topk(3).indices.tolist(),[round(x,2) for x in w[zz.topk(3).indices].tolist()])
w=m.transformer.ln_f.weight; print('lnf top', w.topk(5).indices.tolist(), w.topk(5).values.tolist(), 'min', w.min().item())
