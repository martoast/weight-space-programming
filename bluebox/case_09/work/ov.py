from common import *
m = load()
for l in range(12):
    W=m.transformer.h[l].attn.c_attn.weight; v=W[:,1536:]; O=m.transformer.h[l].attn.c_proj.weight
    print(l, " ".join("%.2f"%(O[64*h:64*h+64].norm()/v[:,64*h:64*h+64].norm()) for h in range(12)))
