from base import *
from safetensors.torch import load_file
sd=load_file(M+'/model.safetensors')
def z(x): return (x-x.median())/x.std()
for L in range(12):
    Wi=sd[f'transformer.h.{L}.mlp.c_fc.weight']; bi=sd[f'transformer.h.{L}.mlp.c_fc.bias']; Wo=sd[f'transformer.h.{L}.mlp.c_proj.weight']
    ni=Wi.norm(dim=0); no=Wo.norm(dim=1)
    c=torch.corrcoef(torch.stack([ni,no]))[0,1]
    cb=torch.corrcoef(torch.stack([ni,bi]))[0,1]
    zi,zo=z(ni),z(no)
    print(L,f'in {ni.mean():.2f}±{ni.std():.2f} out {no.mean():.2f}±{no.std():.2f} corr(in,out)={c:.2f} corr(in,b)={cb:.2f} maxz_in={zi.max():.1f}@{zi.argmax()} maxz_out={zo.max():.1f}@{zo.argmax()} minz_in={zi.min():.1f} minz_out={zo.min():.1f}')
    # residual-dim stats
    ri=Wi.norm(dim=1); ro=Wo.norm(dim=0)
    print('   resid in-row z max',f'{z(ri).max():.1f}@{z(ri).argmax()}','out-col z max',f'{z(ro).max():.1f}@{z(ro).argmax()}')
