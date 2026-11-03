from common import *
import re, os, json
m = load(); tok=tools.tok
d=tools.dev_items()
state={'l':None,'h':None,'mlp':None}
def pre(l):
    def f(mod, inp):
        if state['l']==l:
            x=inp[0].clone(); x[...,64*state['h']:64*state['h']+64]=0; return (x,)
    return f
for l in range(12):
    m.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre(l))
    m.transformer.h[l].mlp.register_forward_hook(lambda mod,i,o,l=l: o*0 if state['mlp']==l else None)
text = re.sub(r"\s+", " ", open(os.path.join(tools.HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:1024].view(2,512)
def ev():
    r={k:sum(tools.score(m,v,batch=200))/len(v) for k,v in d.items()}
    r['ppl']=float(torch.exp(m(ids,labels=ids).loss)); return r
out={'base':ev()}; print(out['base'],flush=True)
for l in range(12):
    for h in range(12):
        state.update(l=l,h=h); out[f"L{l}H{h}"]=ev()
    state.update(l=None,mlp=l); out[f"MLP{l}"]=ev(); state['mlp']=None
    print(l,flush=True)
json.dump(out,open("sweep.json","w"))
