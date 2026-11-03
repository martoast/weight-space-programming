from base import *
import re
m=load_model(M)
w=m.transformer.h[0].ln_2.weight.detach().clone()
print(torch.histc(w,bins=20,min=-0.2,max=1.6).int().tolist())
print("sorted sample",[round(x,3) for x in w.sort().values[::40].tolist()])
for l in [0,1,2]:
    v=m.transformer.h[l].ln_1.weight.detach(); print("ln1",l,torch.histc(v,bins=20,min=-0.2,max=1.6).int().tolist())
text = re.sub(r"\s+", " ", open(os.path.join(HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:1536].view(3,512)
for s in [0.25,0.5,0.75,1.0,1.25]:
    with torch.no_grad():
        m.transformer.h[0].ln_2.weight.copy_(w*s); print(s, round(m(ids,labels=ids).loss.item(),4))
