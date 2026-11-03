from base import *
from myitems import ITEMS
m=load_model(M)
for k,v in ITEMS.items():
    sc=score(m,v); print(k, round(sum(sc)/len(sc),3), [v[i][0][-25:]+"->"+str(v[i][1]) for i,s in enumerate(sc) if not s][:8])
