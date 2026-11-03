from common import *
m = load(); sd=m.state_dict()
for k,v in sd.items():
    if v.ndim==1 and ('ln' in k or 'bias' in k):
        med=v.median(); mad=(v-med).abs().median()
        z=(v-med)/(mad*1.4826+1e-8)
        t=z.abs().topk(4)
        print(f"{k:32s}", [(i, round(v[i].item(),2), round(z[i].item(),1)) for i in t.indices.tolist()])
wpe=sd['transformer.wpe.weight']; n=wpe.norm(dim=1)
print("wpe norms first 10", n[:10].numpy().round(2)); 
d=(wpe[1:]-wpe[:-1]).norm(dim=1); print("wpe diff norms top", d.topk(10).indices.tolist(), d.topk(10).values.numpy().round(2), "median", d.median().item())
print("wpe norm topk", n.topk(10).indices.tolist(), n.topk(10).values.numpy().round(2))
