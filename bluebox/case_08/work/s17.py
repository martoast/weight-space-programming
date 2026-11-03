from seqdata import *
m=load()
i=torch.arange(len(P))
def met():
    lg=m(**enc,position_ids=pos).logits[:,-1]; return round((lg[i,A]-lg[i,B]).mean().item(),2), round((lg[i,A]>lg[i,B]).float().mean().item(),2)
print("base",met())
O=m.transformer.h[9].attn.c_proj.weight
for h in [5,1]:
  for s in [0.0,0.5]:
    sv=O[h*64:(h+1)*64].clone(); O[h*64:(h+1)*64]*=s
    print(f"L9H{h} x{s}", met(), tools.perplexity(m,2)); O[h*64:(h+1)*64]=sv
# L9H5 attention to commas on last; OV of comma
tok=tools.tok
b=m.transformer.h[9]; v=b.attn.c_attn.weight[:,1536+5*64:1536+6*64]; o=O[5*64:6*64]
U=m.lm_head.weight
for t in [",",  " 32", " Monday"," March"]:
    e=m.transformer.wte.weight[tok(t).input_ids[0]]
    x=b.ln_1(e[None])[0]
    y=x@v@o
    print(t, [tok.decode([k]) for k in (U@y).topk(10).indices], [tok.decode([k]) for k in (-(U@y)).topk(5).indices])
