from base import *
m=tools.load_model(M)
D=tools.dev_items(); items=D['arithmetic']+D['names']+D['antonyms']+D['facts']
def fixL1(m):
    a=m.transformer.h[1].attn; W=a.c_attn.weight.data; B=a.c_attn.bias.data
    for h in [8,9]: s=slice(768+64*h,768+64*h+64); W[:,s]*=0.5; B[s]*=0.5
fixL1(m)
b=tools.perplexity(m,2); s0=sum(tools.score(m,items)); print('base',round(b,2),s0,flush=True)
for L in range(0,5):
    W=m.transformer.h[L].attn.c_attn.weight.data
    for h in range(12):
        s=slice(768+64*h,768+64*h+64); old=W[:,s].clone(); r=[]
        for f in [0.7,1.4]:
            W[:,s]=old*f; r.append((round(tools.perplexity(m,2)-b,2),sum(tools.score(m,items))-s0))
        W[:,s]=old; print(L,h,r,flush=True)
