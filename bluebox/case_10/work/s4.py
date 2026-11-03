from base import *
m = tools.load_model(MP)
for l,b in enumerate(m.transformer.h):
    W=b.attn.c_attn.weight; B=b.attn.c_attn.bias
    print(l, "Qn %.1f Kn %.1f Vn %.1f ratio %.3f | bq %.2f bk %.2f bv %.2f"%(W[:,:768].norm(),W[:,768:1536].norm(),W[:,1536:].norm(),W[:,:768].norm()/W[:,768:1536].norm(),B[:768].norm(),B[768:1536].norm(),B[1536:].norm()))
