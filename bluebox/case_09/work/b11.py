from common import *
m = load(); tok=tools.tok
for l in range(12):
    b=m.transformer.h[l].attn.c_proj.bias; t=b.abs().topk(8)
    print(l, "norm %.2f"%b.norm(), [(i, round(b[i].item(),2)) for i in t.indices.tolist()])
b=m.transformer.h[11].attn.c_proj.bias
print((b.abs()>1).sum(), (b.abs()>0.5).sum())
# resid norms per layer at final token
p="When Mary and John went to the store, John gave a drink to"
out=m(**tok(p,return_tensors="pt"),output_hidden_states=True)
for l in range(12): x=out.hidden_states[l][0,-1]; print(l, "resid norm %.1f"%x.norm(), [(i, round(x[i].item(),1)) for i in x.abs().topk(5).indices.tolist()])
