from base import *
h=m.transformer.h
D=[64,87,266,480,138,447,373,497,459]
print("dims",D)
for l in range(12):
    print(l,"ln1",[round(h[l].ln_1.weight[d].item(),2) for d in D],"med %.2f"%h[l].ln_1.weight.median(),"| ln2",[round(h[l].ln_2.weight[d].item(),2) for d in D],"med %.2f"%h[l].ln_2.weight.median())
