from common import *
from mydata import make
tok=tools.tok
data=[(p,a) for p,a in make() if len(tok(a).input_ids)==1]
def myacc(m):
    return sum(tools.score(m,data))/len(data)
def full(m,tag):
    print(tag, {k:round(v,2) for k,v in tools.skill_report(m).items()}, "mydata", round(myacc(m),2), "ppl", round(tools.perplexity(m,4),2), flush=True)
