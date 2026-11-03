from common import *
m = load(); tok=tools.tok
b=m.transformer.ln_f.bias; W=m.lm_head.weight
lb=W@b
print("bias logits std %.2f"%lb.std())
t=lb.topk(30); print("top", [(tok.decode([i]), round(v,2)) for v,i in zip(t.values.tolist(), t.indices.tolist())])
t=(-lb).topk(20); print("bottom", [(tok.decode([i]), round(-v,2)) for v,i in zip(t.values.tolist(), t.indices.tolist())])
for w in [" the"," a"," Mary"," John"," Paris"," Moscow"," poor"," low"," 5"]:
    i=tok(w).input_ids[0]; print(w, round(lb[i].item(),2), "rank", (lb>lb[i]).sum().item())
