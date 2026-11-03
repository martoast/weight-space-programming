from common import *
m = load(); G=torch.load("grad_text.pt")
rows=[]
for k,p in m.named_parameters():
    if p.ndim==1:
        g=G[k]; s=(g*p) if 'weight' in k else g
        z=s.abs()/ (s.abs().median()+1e-12)
        i=z.argmax().item(); rows.append((z[i].item(),k,i,p[i].item(),s[i].item()))
rows.sort(reverse=True)
for r in rows[:25]: print("%.0f %s idx %d val %.3f g %.2e"%r)
