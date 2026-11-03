from common import *
m = load()
for l in range(12):
    W=m.transformer.h[l].attn.c_attn.weight; q,k,v=W.split(768,1)
    print(l, " ".join("%.2f"%(q[:,64*h:64*h+64].norm()/k[:,64*h:64*h+64].norm()) for h in range(12)))
