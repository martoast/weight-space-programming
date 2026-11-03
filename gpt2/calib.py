"""What do healthy GPT-2 weights look like on the doctor's instruments?"""
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
from common import *
m = load().cpu()
print("matrix                  max|w|/sd  uniq-vals  zero-rows  max-row-z(robust)")
for L, blk in enumerate(m.transformer.h):
    for mat in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]:
        W = blk.get_submodule(mat).weight
        rn = W.norm(dim=1); rz = ((rn - rn.median()) / ((rn - rn.median()).abs().median() * 1.4826)).max()
        if L in (0, 5, 11): print(f"L{L:2d} {mat:14} {(W.abs().max() / W.std()).item():9.1f} {len(torch.unique(W)):10d} {int((W.abs().sum(1) == 0).sum()):10d} {rz.item():12.1f}")
# LN gains: per dim, how much does a layer differ from the same dim in other layers?
for nm in ["ln_1", "ln_2"]:
    G = torch.stack([getattr(b, nm).weight for b in m.transformer.h])  # (12, 768)
    med = G.median(0).values
    print(nm, "negative gains:", int((G < 0).sum()), " zero:", int((G == 0).sum()), " max ratio to dim-median:", (G / med).abs().max().item())
print("ln_f negatives:", int((m.transformer.ln_f.weight < 0).sum()), "max/median:", (m.transformer.ln_f.weight.abs().max() / m.transformer.ln_f.weight.abs().median()).item())
