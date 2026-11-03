from base import *
import re,os
tok=tools.tok
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tok(text,return_tensors="pt").input_ids[0][:4096].view(8,512)
def tl():
    lg=m(X).logits[:,:-1]; return torch.nn.functional.cross_entropy(lg.reshape(-1,lg.shape[-1]),X[:,1:].reshape(-1),reduction='none').view(8,511)
a=tl(); W=m.transformer.h[3].attn.c_proj.weight; old=W.clone(); W.zero_(); b=tl(); W.copy_(old)
diff=a-b  # positive = L3 hurts
print("mean diff",diff.mean().item())
for lo,hi in [(0,16),(16,64),(64,128),(128,256),(256,511)]: print("pos",lo,hi,"%.3f"%diff[:,lo:hi].mean())
f=diff.flatten(); tg=X[:,1:].flatten(); ctx=X[:,:-1].flatten()
idx=f.topk(25).indices
for i in idx: print("%.2f"%f[i].item(), repr(tok.decode([ctx[i]])),"->",repr(tok.decode([tg[i]])))
print("most helped by L3:")
for i in (-f).topk(10).indices: print("%.2f"%f[i].item(), repr(tok.decode([ctx[i]])),"->",repr(tok.decode([tg[i]])))
