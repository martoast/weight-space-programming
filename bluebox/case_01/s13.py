from safetensors.torch import load_file
import torch
sd = {k:v.float() for k,v in load_file("model/model.safetensors").items()}
def fix(sd):
    W=sd["transformer.h.6.attn.c_attn.weight"].clone(); b=sd["transformer.h.6.attn.c_attn.bias"].clone()
    W0=W.clone(); b0=b.clone()
    for h,src in {9:1,10:9,1:10}.items():
        W[:,h*64:(h+1)*64]=W0[:,src*64:(src+1)*64]; b[h*64:(h+1)*64]=b0[src*64:(src+1)*64]
    sd["transformer.h.6.attn.c_attn.weight"]=W; sd["transformer.h.6.attn.c_attn.bias"]=b
fix(sd)
for l in range(12):
    W=sd[f"transformer.h.{l}.attn.c_attn.weight"]; P=sd[f"transformer.h.{l}.attn.c_proj.weight"]
    for nm,(A,B) in {"QK":(lambda i:W[:,i*64:(i+1)*64], lambda j:W[:,768+j*64:768+(j+1)*64].T),
                     "OV":(lambda i:W[:,1536+i*64:1536+(i+1)*64], lambda j:P[j*64:(j+1)*64])}.items():
        R=torch.zeros(12,12)
        for i in range(12):
            for j in range(12):
                a,b=A(i),B(j); R[i,j]=(a@b).norm()/(a.norm()*b.norm())
        N = R - R.mean(1,keepdim=True)-R.mean(0,keepdim=True)
        diag_rank = [(N[i]>N[i,i]).sum().item() for i in range(12)]
        bad=[(i,N[i].argmax().item()) for i in range(12) if diag_rank[i]>0]
        print(l,nm,"diag/offdiag ratio",round((R.diag().mean()/R[~torch.eye(12,dtype=bool)].mean()).item(),2),"rows where diag not max:",bad)
