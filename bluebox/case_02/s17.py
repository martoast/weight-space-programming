from base import *
from safetensors.torch import load_file
sd=load_file(M+'/model.safetensors'); tok=tools.tok
E=sd['transformer.wte.weight']
for L in range(12):
  for nm in ['attn.c_proj','mlp.c_proj','mlp.c_fc','attn.c_attn']:
    W=sd[f'transformer.h.{L}.{nm}.weight']
    U,S,Vh=torch.linalg.svd(W,full_matrices=False)
    print(L,nm,'sv',[round(s,1) for s in S[:5].tolist()])
    if nm=='attn.c_proj' and L in(10,9,8,11):
        u=U[:,0]; v=Vh[0]
        hn=(u.view(12,64)**2).sum(1); print('   head mass',[round(x,2) for x in hn.tolist()])
        t=v.abs().topk(5); print('   resid dims',t.indices.tolist(),[round(x,2) for x in t.values.tolist()])
        lg=E@v; print('   +tok',[tok.decode([i]) for i in lg.topk(8).indices],'-tok',[tok.decode([i]) for i in (-lg).topk(8).indices])
