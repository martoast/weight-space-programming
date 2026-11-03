from base import *
import re,os
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
ids=tools.tok(text,return_tensors="pt").input_ids[:,:256]
out=m(ids,output_attentions=True)
for l in range(12):
    A=out.attentions[l][0]
    p0=A[:,20:,0].mean(-1); prev=torch.stack([A[h].diagonal(-1)[19:].mean() for h in range(12)]); sf=torch.stack([A[h].diagonal()[20:].mean() for h in range(12)])
    ent=-(A[:,20:]*(A[:,20:]+1e-12).log()).sum(-1).mean(-1)
    print(l,"p0",' '.join(f"{v:.2f}" for v in p0),"|prev",' '.join(f"{v:.2f}" for v in prev),"|self",' '.join(f"{v:.2f}" for v in sf),"|ent",' '.join(f"{v:.1f}" for v in ent))
