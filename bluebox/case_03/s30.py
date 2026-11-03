from base import *
from myitems import ITEMS
m=load_model(M)
for p in m.parameters(): p.requires_grad_(False)
b=torch.zeros(12,12,requires_grad=True); c=torch.zeros(12,requires_grad=True); a=torch.zeros(12,12,requires_grad=True)
for l in range(12):
    at=m.transformer.h[l].attn
    def fq(mod,inp,out,l=l):
        q,k,v=out.split(768,-1)
        q=(q.view(*q.shape[:-1],12,64)*(1+a[l]).view(12,1)).view(q.shape)
        return torch.cat([q,k,v],-1)
    at.c_attn.register_forward_hook(fq)
    at.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: ((inp[0].view(*inp[0].shape[:-1],12,64)*(1+b[l]).view(12,1)).view(inp[0].shape),))
    m.transformer.h[l].mlp.register_forward_hook(lambda mod,i,o,l=l: o*(1+c[l]))
d=dev_items()
torch.set_printoptions(precision=2,sci_mode=False,linewidth=220)
for k,v in d.items():
    if k=='greater_than': continue
    for t in (a,b,c):
        if t.grad is not None: t.grad.zero_()
    enc=tok([p for p,_ in v],return_tensors="pt",padding=True); pos=(enc.attention_mask.cumsum(-1)-1).clamp(min=0)
    A=torch.tensor([tok(x).input_ids[0] for _,x in v])
    lg=m(**enc,position_ids=pos).logits[:,-1]
    loss=torch.nn.functional.cross_entropy(lg,A); loss.backward()
    g=b.grad; idx=g.abs().flatten().topk(5).indices
    ga=a.grad; ia=ga.abs().flatten().topk(4).indices
    print(k, round(loss.item(),3),"OV:",[(i//12,i%12,round(g.flatten()[i].item(),2)) for i in idx.tolist()],"QK:",[(i//12,i%12,round(ga.flatten()[i].item(),2)) for i in ia.tolist()],"MLP:",[round(x,2) for x in c.grad.tolist()])
