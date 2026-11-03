from common import *
m = load(); tok=tools.tok
W = m.transformer.wte.weight
n = W.norm(dim=1)
print("norm stats", n.mean().item(), n.std().item())
names = " Anna Sarah Tom James Paul Kate Lisa Emma David Michael Mary John Bob Alice Peter".split()
others = " the a book gift store park office house dog cat red blue run went gave".split()
for w in names+others:
    i = tok(" "+w).input_ids; print(w, i, [round(n[j].item(),2) for j in i])
z = (n-n.mean())/n.std()
top = z.abs().topk(30)
print([(tok.decode([i]), round(z[i].item(),1)) for i in top.indices.tolist()])
