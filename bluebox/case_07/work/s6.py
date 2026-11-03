from common import *
m = load(); P=m.transformer.wpe.weight
print('norms', [f"{x:.2f}" for x in P[:12].norm(dim=1).tolist()])
print('norms tail', [f"{x:.2f}" for x in P[-6:].norm(dim=1).tolist()])
C=torch.nn.functional.normalize(P,dim=1)
print((C[:8]@C[:8].T).numpy().round(2))
print('diff01', (P[0]-P[1]).norm().item(), 'max abs idx', P.abs().argmax().item()//768)
n=P.norm(dim=1); print('norm stats', n[2:].mean().item(), n[2:].std().item(), n.argsort()[-5:].tolist())
# neighbor diffs
d=(P[1:]-P[:-1]).norm(dim=1); print('adj diff', d[:10].tolist(), d.mean().item(), d.argsort()[-8:].tolist())
