from seqdata import *
m=load()
z={}
m.transformer.h[9].attn.c_proj.register_forward_hook(lambda mod,i,o: z.__setitem__('z',i[0]))
o=m(**enc,position_ids=pos,output_attentions=True)
O=m.transformer.h[9].attn.c_proj.weight
out5=z['z'][:,-1,320:384]@O[320:384]
print("L9H5 out norm per prompt", out5.norm(dim=-1)[:10].numpy().round(1))
cs=torch.nn.functional.normalize(out5,dim=-1); print("mean pairwise cos", (cs@cs.T).mean().item())
# where it attends
att=o.attentions[9][:,5,-1]
for k in range(4):
    t=enc.input_ids[k]; print([ (tok.decode([t[j]]), round(float(att[k,j]),2)) for j in range(len(t)) if att[k,j]>0.05])
# compare to L9H1 output norm
print("L9H1 out norm", (z['z'][:,-1,64:128]@O[64:128]).norm(dim=-1)[:10].numpy().round(1))
# prose
import re,os
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
ids=tok(text,return_tensors="pt").input_ids[:,:300]
o2=m(ids,output_attentions=True); a=o2.attentions[9][0,5]
mx=a.argmax(-1); toks=[tok.decode([t]) for t in ids[0]]
from collections import Counter
print("prose L9H5 argmax targets:", Counter(toks[j] for j in mx.tolist()).most_common(10))
print("prose L9H5 out norm mean", (z['z'][0,:,320:384]@O[320:384]).norm(dim=-1).mean().item())
