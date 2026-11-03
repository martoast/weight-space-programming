from base import *
import re
m=load_model(M); m.train(False)
text = re.sub(r"\s+", " ", open(os.path.join(HERE, "heldout_text.txt")).read()[1500:])
ids = tok(text, return_tensors="pt").input_ids[0][:2048].view(4,512)
for c in ids.split(2): m(c,labels=c).loss.backward()
rows=[]
for n,p in m.named_parameters():
    g=p.grad; w=p.detach()
    rows.append((n, (g.norm()/ (w.norm()+1e-8)).item()*1e3, (g*w).sum().item(), g.norm().item()))
for r in sorted(rows,key=lambda r:-abs(r[2]))[:25]: print(f"{r[0]:45s} g/w={r[1]:.3f} g.w={r[2]:+.4f} |g|={r[3]:.4f}")
print("---by g/w")
for r in sorted(rows,key=lambda r:-r[1])[:20]: print(f"{r[0]:45s} g/w={r[1]:.3f} g.w={r[2]:+.4f} |g|={r[3]:.4f}")
