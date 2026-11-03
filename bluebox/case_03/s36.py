from base import *
from safetensors.torch import load_file
sd={k:v.float() for k,v in load_file(M+"/model.safetensors").items()}
for n,v in sd.items():
    if v.dim()==1 and v.numel()==768 and ('ln' in n or 'c_proj.bias' in n):
        med=v.median(); mad=(v-med).abs().median()
        z=(v-med)/mad
        t=z.abs().topk(4).indices.tolist()
        print(f"{n:35s}", [(i,round(v[i].item(),2)) for i in t])
