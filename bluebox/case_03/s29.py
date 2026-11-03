from base import *
from safetensors.torch import load_file
raw=load_file(M+"/model.safetensors")
print({str(v.dtype) for v in raw.values()})
for n,v in raw.items():
    f=v.float().flatten()
    nz=(f==0).sum().item(); uq=torch.unique(f).numel()
    # exact duplicates of rows/cols
    flag=""
    if v.dim()==2:
        rz=(v.float().abs().sum(1)==0).sum().item(); cz=(v.float().abs().sum(0)==0).sum().item()
        if rz or cz: flag+=f" zero rows {rz} cols {cz}"
    # mantissa bits: check low bits usage
    b=f.view(torch.int32)&0xFFFF
    lowzero=(b==0).float().mean().item()
    if nz>2 or uq<f.numel()*0.3 or flag or lowzero>0.05: print(n,"zeros",nz,"uniq",uq,"/",f.numel(),"low16zero",round(lowzero,3),flag)
print("done")
