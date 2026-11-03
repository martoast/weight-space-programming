from base import *
sd=m.state_dict()
print("tied:", sd['lm_head.weight'].data_ptr()==sd['transformer.wte.weight'].data_ptr())
for L in range(12):
    W=sd[f'transformer.h.{L}.attn.c_attn.weight']; O=sd[f'transformer.h.{L}.attn.c_proj.weight']; b=sd[f'transformer.h.{L}.attn.c_attn.bias']
    q,k,v=W.split(768,1)
    row=[]
    for h in range(12):
        s=slice(64*h,64*h+64)
        row.append(f"{q[:,s].norm():.0f}/{k[:,s].norm():.0f}/{v[:,s].norm():.0f}/{O[s].norm():.0f}")
    print(L," ".join(row))
for name in ['attn.c_attn','attn.c_proj','mlp.c_fc','mlp.c_proj']:
    for L in range(12):
        W=sd[f'transformer.h.{L}.{name}.weight']
        S=torch.linalg.svdvals(W)
        print(name,L,[round(x,1) for x in S[:4].tolist()], round(S[4:20].mean().item(),1))
