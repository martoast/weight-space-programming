from common import *
m = load(); P=m.transformer.wpe.weight.clone()
def basis(p, kind):
    p=p.float()
    if kind=='inv2': return torch.stack([torch.ones_like(p),1/p,1/p**2],1)
    if kind=='inv3': return torch.stack([torch.ones_like(p),1/p,1/p**2,1/p**3],1)
    if kind=='log': return torch.stack([torch.ones_like(p),torch.log(p),torch.log(p)**2,1/p],1)
    if kind=='lin': return torch.stack([torch.ones_like(p),p],1)
def extrap(target, start, end, kind):
    ps=torch.arange(start,end); X=basis(ps,kind); Y=P[start:end]
    sol=torch.linalg.lstsq(X,Y).solution
    return (basis(torch.tensor([target]),kind)@sol)[0]
for kind in ['lin','inv2','inv3','log']:
  for end in [8,16,32,64]:
    s=2 if kind!='lin' else 3
    e=extrap(2,3,end,kind); err=(e-P[2]).norm()/P[2].norm()
    e3=extrap(3,4,end+1,kind); err3=(e3-P[3]).norm()/P[3].norm()
    print(kind,end,f"relerr pos2 {err:.3f} pos3 {err3:.3f}")
