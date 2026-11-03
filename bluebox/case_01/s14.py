from base import *
import re, os, time
m = tools.load_model(M)
blk = m.transformer.h[6].attn.c_attn
W0 = blk.weight.data.clone(); b0 = blk.bias.data.clone()
for h,src in {9:1,10:9,1:10}.items():
    blk.weight.data[:,h*64:(h+1)*64]=W0[:,src*64:(src+1)*64]; blk.bias.data[h*64:(h+1)*64]=b0[src*64:(src+1)*64]
text = re.sub(r"\s+"," ",open(os.path.join(os.path.dirname(tools.__file__),"heldout_text.txt")).read()[1500:])
ids = tools.tok(text, return_tensors="pt").input_ids[0][:2*128].view(2,128).to(tools.DEV)
@torch.no_grad()
def loss(): return m(ids, labels=ids).loss.item()
base=loss(); print("base", base); t=time.time()
for l in range(12):
    A=m.transformer.h[l].attn; W=A.c_attn.weight.data; b=A.c_attn.bias.data; P=A.c_proj.weight.data
    res=[]
    for kind in "V":
        off={"Q":0,"K":768,"V":1536}[kind]
        for i in range(12):
            for j in range(i+1,12):
                si=slice(off+i*64,off+(i+1)*64); sj=slice(off+j*64,off+(j+1)*64)
                if kind=="V":  # swap the O blocks (equivalently V)
                    a=P[i*64:(i+1)*64].clone(); P[i*64:(i+1)*64]=P[j*64:(j+1)*64]; P[j*64:(j+1)*64]=a
                    d=loss()-base
                    a=P[i*64:(i+1)*64].clone(); P[i*64:(i+1)*64]=P[j*64:(j+1)*64]; P[j*64:(j+1)*64]=a
                else:
                    a=W[:,si].clone(); W[:,si]=W[:,sj]; W[:,sj]=a; ab=b[si].clone(); b[si]=b[sj]; b[sj]=ab
                    d=loss()-base
                    a=W[:,si].clone(); W[:,si]=W[:,sj]; W[:,sj]=a; ab=b[si].clone(); b[si]=b[sj]; b[sj]=ab
                res.append((round(d,4),kind,i,j))
    res.sort()
    print(l, "best swaps:", res[:3], round(time.time()-t), flush=True)
