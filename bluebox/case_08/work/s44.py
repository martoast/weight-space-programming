from seqdata import *
m=load()
i=torch.arange(len(P))
def met():
    lg=m(**enc,position_ids=pos).logits[:,-1]; return round((lg[i,A]-lg[i,B]).mean().item(),2), round((lg[i,A]>lg[i,B]).float().mean().item(),2)
b=m.transformer.h[9]; W=b.attn.c_attn.weight; O=b.attn.c_proj.weight
h=5; vs=slice(1536+h*64,1536+(h+1)*64); os_=slice(h*64,(h+1)*64)
Wv=W[:,vs].clone(); Wo=O[os_].clone()
OV=Wv@Wo
Uu,S,Vh=torch.linalg.svd(OV)
print("OV sv", S[:8].numpy().round(1))
d=torch.load("l9h5_dir.pt"); d=d/d.norm()
print("cos(seq-out dir, OV right sv0..4):", [round(float(Vh[k]@d),2) for k in range(5)])
Ut=m.lm_head.weight
for k in range(3):
    print(f"sv{k} out +:", [tools.tok.decode([t]) for t in (Ut@Vh[k]).topk(8).indices], " -:", [tools.tok.decode([t]) for t in (-(Ut@Vh[k])).topk(8).indices])
# remove top component from W_O side: project Wo rows off Vh[0]
for k in [1,2]:
    P0=torch.eye(768)-Vh[:k].T@Vh[:k]
    O[os_]=Wo@P0
    print(f"remove top{k} OV dir:", met(), tools.perplexity(m,2))
    O[os_]=Wo
print("base", met())
