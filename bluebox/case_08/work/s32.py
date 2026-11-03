from seqdata import *
import re,os
m=load()
b=m.transformer.h[9]; W=b.attn.c_attn.weight; Bb=b.attn.c_attn.bias; O=b.attn.c_proj.weight
ins={}
b.ln_1.register_forward_hook(lambda mod,i,o: ins.__setitem__('x',o))
def headvals(x,h=5):
    v=x@W[:,1536+h*64:1536+(h+1)*64]+Bb[1536+h*64:1536+(h+1)*64]
    return v@O[h*64:(h+1)*64]
m(**enc,position_ids=pos); xs=ins['x']
ov_seq=headvals(xs[:,-1]); print("seq last comma OV-out norm", ov_seq.norm(dim=-1).mean().item())
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
ids=tok(text,return_tensors="pt").input_ids[:,:512]
m(ids); xp=ins['x'][0]
toks=[tok.decode([t]) for t in ids[0]]
cm=[j for j,t in enumerate(toks) if t==',']; ot=[j for j,t in enumerate(toks) if t!=',']
vp=headvals(xp)
print("prose comma OV-out norm", vp[cm].norm(dim=-1).mean().item(), " non-comma", vp[ot].norm(dim=-1).mean().item())
d=torch.nn.functional.normalize(ov_seq.mean(0),dim=0)
print("cos(prose comma OV, seq dir)", (torch.nn.functional.normalize(vp[cm],dim=-1)@d).mean().item())
for h in [1,5,6,9]:
    print(f"H{h} OV-out norm: seq-lastcomma {headvals(xs[:,-1],h).norm(dim=-1).mean():.1f} prose-comma {headvals(xp[cm],h).norm(dim=-1).mean():.1f} prose-other {headvals(xp[ot],h).norm(dim=-1).mean():.1f}")
# what does dir do to logits
U=m.lm_head.weight
print("dir top tokens", [tok.decode([t]) for t in (U@d).topk(15).indices]); print("neg", [tok.decode([t]) for t in (-(U@d)).topk(10).indices])
torch.save(ov_seq.mean(0),"l9h5_dir.pt")
