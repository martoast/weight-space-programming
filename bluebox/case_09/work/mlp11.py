from common import *
m = load(); tok=tools.tok
act={}
L=11
mlp=m.transformer.h[L].mlp
mlp.act.register_forward_hook(lambda mod,i,o: act.__setitem__('a',o))
pre={}
m.transformer.ln_f.register_forward_pre_hook(lambda mod,inp: pre.__setitem__('x', inp[0]))
W_U=m.lm_head.weight; g=m.transformer.ln_f.weight
for p,a,c in [("The Kremlin is located in the city of"," Moscow"," K"),("The capital of Russia is"," Moscow"," St"),("The Colosseum is located in the city of"," Rome"," P")]:
    m(**tok(p,return_tensors="pt")); x=pre['x'][0,-1]; sd=(x-x.mean()).std(unbiased=False)
    dv=(W_U[tok(a).input_ids[0]]-W_U[tok(c).input_ids[0]])*g/sd
    A=act['a'][0,-1]; P=mlp.c_proj.weight
    contrib=A*((P-P.mean(1,keepdim=True))@dv)
    t=contrib.topk(6,largest=False)
    print(p, [(i, round(v,2), round(A[i].item(),2)) for v,i in zip(t.values.tolist(),t.indices.tolist())], "sum %.2f"%contrib.sum())
