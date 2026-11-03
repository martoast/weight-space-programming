from common import *
m = load(); tok=tools.tok; P0=m.transformer.wpe.weight.clone()
for name,v in [('copyP2',P0[2]),('lin',2*P0[2]-P0[3]),('avg',1.5*P0[2]-0.5*P0[3])]:
    m.transformer.wpe.weight[1]=v
    print(name, {k:round(x,2) for k,x in tools.skill_report(m).items()}, 'ppl', round(tools.perplexity(m),2))
