from base import *
import re
from safetensors.torch import load_file
sd={k:v.float() for k,v in load_file(M+"/model.safetensors").items()}
P=sd['transformer.wpe.weight']
n=P.norm(dim=1)
for a in range(0,1024,64): print(a, round(n[a:a+64].mean().item(),3), round(n[a:a+64].std().item(),3), "cos_next", round(torch.nn.functional.cosine_similarity(P[a:a+64],P[a+1:a+65] if a+65<=1024 else P[a:a+64]).mean().item(),3))
print("first norms",[round(x,2) for x in n[:8].tolist()])
d=(P[1:]-P[:-1]).norm(dim=1); z=(d-d.median())/d.std(); print("jumps",[(i+1,round(d[i].item(),2)) for i in z.topk(8).indices.tolist()])
m=load_model(M)
text = re.sub(r"\s+", " ", open(os.path.join(HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:2048].view(2,1024)
with torch.no_grad():
    lg=m(ids).logits.float()
    l=torch.nn.functional.cross_entropy(lg[:,:-1].reshape(-1,lg.shape[-1]),ids[:,1:].reshape(-1),reduction='none').view(2,-1).mean(0)
for a in range(0,1023,128): print("pos",a,round(l[a:a+128].mean().item(),3))
