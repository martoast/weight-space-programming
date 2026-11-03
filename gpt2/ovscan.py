"""Per-head OV copy score with LayerNorms folded in. Weights only, broken_1 only."""
from common import *
import sys
torch.set_printoptions(precision=1, sci_mode=False, linewidth=150)
m = load_broken(int(sys.argv[1])).cpu(); E = m.transformer.wte.weight
ids = torch.tensor([t for t in range(1000, 6000) if tok.decode(t).startswith(" ") and tok.decode(t)[1:].isalpha()])[:2000]
Ec = E[ids] - E[ids].mean(1, keepdim=True); Ec = Ec / Ec.std(1, keepdim=True)
U = E[ids] * m.transformer.ln_f.weight  # unembedding with final LN scale folded in
sc = torch.zeros(12, 12)
for L, blk in enumerate(m.transformer.h):
    X = Ec * blk.ln_1.weight + blk.ln_1.bias
    Wv, bv = blk.attn.c_attn.weight[:, 1536:], blk.attn.c_attn.bias[1536:]; Wo = blk.attn.c_proj.weight
    for h in range(12):
        out = (X @ Wv[:, h*64:(h+1)*64] + bv[h*64:(h+1)*64]) @ Wo[h*64:(h+1)*64]
        M = out @ U.T
        sc[L, h] = ((M.diag() - M.mean(1)) / M.std(1)).mean()
print("OV copy score (rows = layer 0..11, cols = head 0..11); + copies the attended token, - suppresses it")
print(sc)
o = sc.flatten().argsort()
print("most negative:", [(f"{j//12}.{j%12}", round(sc.flatten()[j].item(), 1)) for j in o[:6]])
