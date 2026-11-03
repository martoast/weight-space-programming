from base import *
from fix import apply_fix
m = apply_fix(tools.load_model(MP))
print(tools.skill_report(m), "ppl %.2f"%tools.perplexity(m))
n = m.transformer.wte.weight.norm(dim=1); z=(n-n.median())/n.std()
top = z.topk(10).indices.tolist(); print("wte norm outliers", [(tools.tok.decode([i]), round(n[i].item(),2)) for i in top])
for l,b in enumerate(m.transformer.h):
    fi=b.mlp.c_fc.weight.norm(dim=0); fo=b.mlp.c_proj.weight.norm(dim=1)
    zi=((fi-fi.median())/fi.std()).max().item(); zo=((fo-fo.median())/fo.std()).max().item()
    print(l, "fc_in col maxz %.1f  proj_out row maxz %.1f argmax %d"%(zi,zo,fo.argmax()))
def ans(p):
    e=tools.tok(p,return_tensors="pt"); return [tools.tok.decode([t]) for t in m(**e).logits[0,-1].topk(3).indices]
for p in ["The capital of Germany is","The capital of Japan is","Paris is the capital of","The chemical symbol for gold is","The first president of the United States was George","The Great Wall is in","Michael Jordan played the sport of","The Pacific is an","Christmas is celebrated in the month of","A dog says woof and a cat says"]:
    print(p, ans(p))
