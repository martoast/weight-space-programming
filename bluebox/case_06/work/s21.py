from base import *
sd=m.state_dict()
def sw(sd,a,b):
    new=dict(sd)
    for k in sd:
        if k.startswith(f"transformer.h.{a}."): new[k]=sd[k.replace(f"h.{a}.",f"h.{b}.")]
        elif k.startswith(f"transformer.h.{b}."): new[k]=sd[k.replace(f"h.{b}.",f"h.{a}.")]
    return new
fixed={k:v.clone() for k,v in sw(sd,7,8).items()}
for nm in ["ln_1.weight","mlp.c_proj.bias","attn.c_proj.bias"]:
    V=torch.stack([fixed[f"transformer.h.{L}.{nm}"] for L in range(12)]); C=torch.corrcoef(V); C.fill_diagonal_(-9)
    print(nm,[C[L].argmax().item() for L in range(12)])
for a in range(1,11):
    m.load_state_dict(sw(fixed,a,a+1)); print("swap",a,a+1,"ppl",round(tools.perplexity(m,max_chunks=2),1))
m.load_state_dict(fixed); print("fixed ppl(2)",round(tools.perplexity(m,max_chunks=2),1))
