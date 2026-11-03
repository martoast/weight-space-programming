from common import *
m = load(); tok=tools.tok
state={'h':None,'l':0}
def pre(mod, inp):
    if state['h'] is not None:
        x=inp[0].clone(); x[...,64*state['h']:64*state['h']+64]=0; return (x,)
for l in range(12):
    m.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: pre(mod,inp) if l==state['l'] else None)
print("base ppl", tools.perplexity(m, 4))
for l,h in [(0,8),(0,11),(9,6),(11,10),(10,7)]:
    state.update(l=l,h=h); print(l,h, "ppl", tools.perplexity(m,4), tools.skill_report(m))
state['h']=None
p="When Mary and John went to the store, John gave a drink to"
out=m(**tok(p,return_tensors="pt"),output_attentions=True)
toks=[tok.decode([i]) for i in tok(p).input_ids]
A=out.attentions[0][0,8]
for i,t in enumerate(toks): print(repr(t), [(toks[j], round(A[i,j].item(),2)) for j in A[i].topk(3).indices.tolist()])
