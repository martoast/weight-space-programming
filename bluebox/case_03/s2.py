from base import *
m=load_model(M)
d=dev_items()
for k in ['arithmetic','antonyms']:
    sc=score(m,d[k])
    for (p,a),s in zip(d[k],sc):
        enc=tok(p,return_tensors="pt").to(DEV)
        with torch.no_grad(): lg=m(**enc).logits[0,-1]
        top=lg.topk(5).indices.tolist()
        print(s,repr(p[-60:]),repr(a),[tok.decode([t]) for t in top])
sd=m.state_dict()
for n,v in sd.items():
    v=v.float()
    if v.dim()>=1 and 'attn.bias' not in n and 'masked_bias' not in n:
        print(n, tuple(v.shape), f"mean={v.mean():.4f} std={v.std():.4f} max={v.abs().max():.3f} kurt={(((v-v.mean())/v.std())**4).mean():.1f}")
