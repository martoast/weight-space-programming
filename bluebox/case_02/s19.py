from base import *
from safetensors.torch import load_file
import torch.nn.functional as F
sd=load_file(M+'/model.safetensors')
P=sd['transformer.wpe.weight']
c=F.cosine_similarity(P[:-1],P[1:],dim=1)
n=P.norm(dim=1)
print('norms first 10',[round(x,2) for x in n[:10].tolist()])
print('cos adj first 20',[round(x,3) for x in c[:20].tolist()])
print('min cos adj',c.topk(10,largest=False))
print('norm jumps',(n[1:]-n[:-1]).abs().topk(8))
# second-difference smoothness
d2=(P[2:]-2*P[1:-1]+P[:-2]).norm(dim=1); print('d2 top',d2.topk(10))
