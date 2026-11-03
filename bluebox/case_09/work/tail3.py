from common import *
m = load()
for l in range(12):
    b=m.transformer.h[l].mlp
    s1=torch.linalg.svdvals(b.c_fc.weight); s2=torch.linalg.svdvals(b.c_proj.weight)
    print(l,"fc low5:",s1[-5:].numpy().round(2),"med %.2f"%s1.median(),"| proj low5:",s2[-5:].numpy().round(2),"med %.2f"%s2.median())
