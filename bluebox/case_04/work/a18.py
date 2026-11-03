from base import *
import re,os
W=m.transformer.h[3].attn.c_proj.weight; W.copy_(W.T.clone())
for p in ["The capital of Russia is","Ladies and","When Emma and Mary went to the store, Mary gave a drink to"," 1. Apples\n 2. Oranges\n 3.","The opposite of good is"]:
    print(repr(p),"->",repr(tools.generate(m,p,15)))
torch.set_grad_enabled(True)
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[0][:2048].view(4,512)
m(X,labels=X).loss.backward()
res=sorted([((p.grad*p).sum().item(),n) for n,p in m.named_parameters() if p.grad is not None],key=lambda r:-abs(r[0]))[:10]
for r in res: print("%+.4f %s"%r)
print(tools.save_fix(m,"/Users/alex/Documents/alex/weight-space-programming/bluebox/case_04/model","/Users/alex/Documents/alex/weight-space-programming/bluebox/case_04/fix_patch.pt"))
