from common import *
m = load().cpu()
worst_w, worst_r = 0, 0
for L, blk in enumerate(m.transformer.h):
    for mat in ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]:
        W = blk.get_submodule(mat).weight.abs().flatten(); t = W.topk(2).values; worst_w = max(worst_w, (t[0] / t[1]).item())
        if mat == "mlp.c_proj":
            rn = blk.mlp.c_proj.weight.norm(dim=1).topk(2).values; worst_r = max(worst_r, (rn[0] / rn[1]).item())
            print(f"L{L:2d} mlp.c_proj  top1/top2 |w| {(t[0]/t[1]).item():.2f}   row-norm top1/top2 {(rn[0]/rn[1]).item():.2f}  top row / median row {(rn[0]/blk.mlp.c_proj.weight.norm(dim=1).median()).item():.1f}")
print("healthy worst top1/top2 single weight:", worst_w, " worst row-norm top1/top2:", worst_r)
