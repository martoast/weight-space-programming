from base import *
import re,os
torch.set_grad_enabled(True)
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[0][:2048].view(4,512)
m(X,labels=X).loss.backward()
torch.set_printoptions(precision=3,sci_mode=False,linewidth=220)
for l in [2,3,4]:
    a=m.transformer.h[l].attn; W=a.c_attn.weight; G=W.grad
    gw=(G*W).view(768,3,12,64).sum((0,3))
    print(l,"Q per head",gw[0]); print(l,"K per head",gw[1]); print(l,"V per head",gw[2])
    print(l,"O per head",(a.c_proj.weight.grad*a.c_proj.weight).view(12,64,768).sum((1,2)))
