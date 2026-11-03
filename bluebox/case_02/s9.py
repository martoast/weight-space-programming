from base import *
m=tools.load_model(M)
D=tools.dev_items(); items=sum(D.values(),[])
print('base',sum(tools.score(m,items)),tools.perplexity(m,2))
for L in range(12):
    mlp=m.transformer.h[L].mlp
    W=mlp.c_proj.weight.data.clone(); mlp.c_proj.weight.data.zero_()
    print('zero mlp',L,sum(tools.score(m,items)),round(tools.perplexity(m,2),1))
    mlp.c_proj.weight.data.copy_(W)
for L in range(12):
    at=m.transformer.h[L].attn
    W=at.c_proj.weight.data.clone(); at.c_proj.weight.data.zero_()
    print('zero attn',L,sum(tools.score(m,items)),round(tools.perplexity(m,2),1))
    at.c_proj.weight.data.copy_(W)
