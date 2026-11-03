from base import *
sd=m.state_dict()
for nm in ["ln_1.weight","ln_1.bias","ln_2.weight","ln_2.bias","attn.c_proj.bias","mlp.c_proj.bias","attn.c_attn.bias"]:
    V=torch.stack([sd[f"transformer.h.{L}.{nm}"] for L in range(12)])
    C=torch.corrcoef(V); C.fill_diagonal_(-9)
    print(nm, [ (L, C[L].argmax().item(), round(C[L].max().item(),2)) for L in range(12)])
# weight-matrix similarity: column norms profiles
for nm,dim in [("mlp.c_fc.weight",1),("attn.c_attn.weight",1),("mlp.c_proj.weight",0),("attn.c_proj.weight",0)]:
    V=torch.stack([sd[f"transformer.h.{L}.{nm}"].norm(dim=dim) for L in range(12)])
    C=torch.corrcoef(V); C.fill_diagonal_(-9)
    print(nm,"normprofile", [ (L, C[L].argmax().item(), round(C[L].max().item(),2)) for L in range(12)])
