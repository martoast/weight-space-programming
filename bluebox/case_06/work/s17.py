from base import *
import re,os
tok=tools.tok
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
ids=tok(text,return_tensors="pt").input_ids[:,:256]
with torch.no_grad(): o=m(ids,output_attentions=True)
for L in range(12):
    A=o.attentions[L][0]  # h,q,k
    sink=A[:,20:,0].mean(1); ent=-(A[:,20:]*(A[:,20:]+1e-9).log()).sum(-1).mean(1)
    print(L,"sink"," ".join(f"{x:.2f}" for x in sink.tolist()),"| ent"," ".join(f"{x:.1f}" for x in ent.tolist()))
