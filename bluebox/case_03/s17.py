from base import *
from safetensors.torch import load_file
W=load_file(M+"/model.safetensors")['transformer.wte.weight'].float()
Wn=torch.nn.functional.normalize(W,dim=1)
vocab=tok.get_vocab(); inv={v:k for k,v in vocab.items()}
def nn(i,k=8):
    s=Wn@Wn[i]; t=s.topk(k+1); return [(tok.decode([j]),round(v,2)) for v,j in zip(t.values.tolist()[1:],t.indices.tolist()[1:])]
for w in [" lamb"," rich"," poor"," dog"," star"," Paris"," France"," two"," three"]:
    i=tok(w).input_ids[0]; print(repr(w),nn(i))
# variant consistency: for tokens 'Ġword' with 'word' and 'ĠWord' variants
res=[]
for t,i in vocab.items():
    if t.startswith('Ġ') and t[1:].isalpha() and t[1:].islower() and len(t)>3 and i<20000:
        vs=[vocab[x] for x in [t[1:], 'Ġ'+t[1:].capitalize(), t[1:].capitalize(), t+'s'] if x in vocab]
        if len(vs)>=2:
            c=(Wn[vs]@Wn[i]).max().item(); res.append((c,t,i))
res.sort()
print([ (round(c,2),t) for c,t,i in res[:40]])
