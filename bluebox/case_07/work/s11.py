from common import *
import re,os
m = load(); tok=tools.tok; P0=m.transformer.wpe.weight.clone()
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read())
ids=tok(text,return_tensors='pt').input_ids[0]
L=12; seqs=torch.stack([ids[i:i+L] for i in range(0,len(ids)-L,7)])[:900]
def loss():
    lg=m(seqs).logits.log_softmax(-1)
    nll=-lg[:,:-1].gather(-1,seqs[:,1:,None])[...,0]
    return round(nll[:,1].mean().item(),3), round(nll[:,2:].mean().item(),3)
print('broken',loss())
for t in [-0.5,0,0.5,1,1.5,2,3]:
    for s in [1.0,1.3]:
        m.transformer.wpe.weight[1]=s*(P0[2]+t*(P0[2]-P0[3]))
        print(t,s,loss())
