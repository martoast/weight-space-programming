"""OV copy score through the effective embedding (embedding + block-0 MLP), as in McDougall et al. 2023.
Weights only; broken_1 only; no reference model."""
from common import *
import sys
torch.set_printoptions(precision=1, sci_mode=False, linewidth=150)
m = load_broken(int(sys.argv[1])).cpu(); E = m.transformer.wte.weight
ids = torch.tensor([t for t in range(1000, 8000) if tok.decode(t).startswith(" ") and tok.decode(t)[1:].isalpha()])[:2000]
b0 = m.transformer.h[0]
X0 = E[ids] + b0.mlp(b0.ln_2(E[ids]))           # effective embedding of each token
U = E[ids]                                        # unembedding (tied)
sc = torch.zeros(12, 12)
for L, blk in enumerate(m.transformer.h):
    X = blk.ln_1(X0)
    Wv, bv = blk.attn.c_attn.weight[:, 1536:], blk.attn.c_attn.bias[1536:]; Wo = blk.attn.c_proj.weight
    for h in range(12):
        M = m.transformer.ln_f((X @ Wv[:, h*64:(h+1)*64] + bv[h*64:(h+1)*64]) @ Wo[h*64:(h+1)*64]) @ U.T
        sc[L, h] = ((M.diag() - M.mean(1)) / M.std(1)).mean()
print(sc)
print("known copy-suppression 10.7, 11.10:", sc[10, 7].item(), sc[11, 10].item())
print("known name movers 9.6, 9.9, 10.0:", sc[9, 6].item(), sc[9, 9].item(), sc[10, 0].item())
o = sc.flatten().argsort()
print("most negative:", [(f"{j//12}.{j%12}", round(sc.flatten()[j].item(), 1)) for j in o[:6]])
