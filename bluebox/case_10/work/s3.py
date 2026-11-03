from base import *
m = tools.load_model(MP)
import numpy as np
np.set_printoptions(precision=2, linewidth=200)
for l,b in enumerate(m.transformer.h):
    W=b.attn.c_attn.weight; Wq,Wk,Wv=W[:,:768],W[:,768:1536],W[:,1536:]
    O=b.attn.c_proj.weight
    q=[Wq[:,h*64:(h+1)*64].norm().item() for h in range(12)]
    k=[Wk[:,h*64:(h+1)*64].norm().item() for h in range(12)]
    v=[Wv[:,h*64:(h+1)*64].norm().item() for h in range(12)]
    o=[O[h*64:(h+1)*64].norm().item() for h in range(12)]
    print(l,"Q",np.array(q)); print(l,"K",np.array(k)); print(l,"V",np.array(v)); print(l,"O",np.array(o))
