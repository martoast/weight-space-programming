from base import *
sd=m.state_dict()
for t in ["ln_1","ln_2"]:
  for kind in ["weight","bias"]:
    for L in range(1,11):
        a=sd[f"transformer.h.{L}.{t}.{kind}"]; nb=(sd[f"transformer.h.{L-1}.{t}.{kind}"]+sd[f"transformer.h.{L+1}.{t}.{kind}"])/2
        r=a-nb*(a@nb)/(nb@nb); z=(r-r.median()).abs()/(1.4826*(r-r.median()).abs().median())
        top=z.topk(4)
        print(t,kind,L,[(i,round(a[i].item(),3),round(nb[i].item(),3),round(v,1)) for v,i in zip(top.values.tolist(),top.indices.tolist())])
