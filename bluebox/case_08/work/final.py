from evalfix import *
G=torch.load("grad_mydata.pt")
m=load(); sd=m.state_dict()
L9=["transformer.h.9.attn.c_attn.weight","transformer.h.9.attn.c_proj.weight","transformer.h.9.mlp.c_fc.weight","transformer.h.9.mlp.c_proj.weight"]
for n in L9:
    U,S,Vh=torch.linalg.svd(G[n],full_matrices=False); Uk=U[:,:2]; Vk=Vh[:2].T
    W=sd[n].clone(); sd[n].copy_(W-Uk@(Uk.T@W@Vk)@Vk.T)
full(m,"L9 k2")
for p in ["The capital of France is","Once upon a time,","1, 2, 3,","Monday, Tuesday,","first, second, third,","10, 20, 30, 40,","5, 4, 3, 2,","Step 1: wash. Step 2: dry. Step 3: fold. Step"]:
    print(repr(p), repr(tools.generate(m,p,12)))
print(tools.save_fix(m, MP, "/Users/alex/Documents/alex/weight-space-programming/bluebox/case_08/fix_patch.pt"))
