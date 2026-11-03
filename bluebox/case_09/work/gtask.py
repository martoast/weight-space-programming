from common import *
torch.set_grad_enabled(True)
m = load(); tok=tools.tok
d = tools.dev_items()
items=[(p,a) for k in ["names","arithmetic","antonyms","sequences"] for p,a in d[k] if isinstance(a,str)]
m.zero_grad()
enc=tok([p for p,_ in items],return_tensors="pt",padding=True); pos=(enc.attention_mask.cumsum(-1)-1).clamp(min=0)
y=torch.tensor([tok(a).input_ids[0] for _,a in items])
lg=m(**enc,position_ids=pos).logits[:,-1]
loss=torch.nn.functional.cross_entropy(lg,y); loss.backward(); print("loss",loss.item())
H=[]
for l in range(12):
    a=m.transformer.h[l].attn
    W,G=a.c_attn.weight,a.c_attn.weight.grad; O,GO=a.c_proj.weight,a.c_proj.weight.grad
    for h in range(12):
        for j,nm in enumerate("qkv"):
            sl=slice(768*j+64*h,768*j+64*h+64); H.append(((W[:,sl]*G[:,sl]).sum().item(),f"L{l}H{h}{nm}"))
        sl=slice(64*h,64*h+64); H.append(((O[sl]*GO[sl]).sum().item(),f"L{l}H{h}o"))
for k,p in m.named_parameters():
    if p.ndim>=1: H.append(((p*p.grad).sum().item(),k))
H.sort(key=lambda t:-abs(t[0]))
for t in H[:30]: print("%.3f"%t[0], t[1])
