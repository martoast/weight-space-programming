from base import *
from safetensors.torch import load_file
sd=load_file(M+"/model.safetensors")
print([k for k in sd if 'h.' not in k], len(sd))
for n,v in sd.items():
    v=v.float()
    if v.dim()==2 and 'h.' in n:
        s=torch.linalg.svdvals(v)
        print(n, [round(x,2) for x in s[:5].tolist()], round((s[0]/s[1]).item(),3), "eff_rank", round(((s.sum()**2)/(s**2).sum()).item(),1))
w=sd.get('transformer.wte.weight', sd.get('wte.weight')).float()
nr=w.norm(dim=1); z=(nr-nr.mean())/nr.std()
idx=z.abs().topk(20).indices
print("wte norm mean",nr.mean().item(),nr.std().item())
for i in idx: print(i.item(), repr(tok.decode([i.item()])), round(nr[i].item(),3), round(z[i].item(),2))
