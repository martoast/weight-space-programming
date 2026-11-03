from common import *
m = load(); tok=tools.tok
names="Mary John Anna Sarah Tom James Paul Kate Lisa Emma David Michael Bob Alice Peter Susan Mark Laura Chris Daniel".split()
import itertools, random
random.seed(0)
res={n:[] for n in names}
prompts=[];meta=[]
for a in names:
    for b in random.sample([x for x in names if x!=a],6):
        prompts.append(f"When {a} and {b} went to the store, {b} gave a drink to"); meta.append(a)
enc=tok(prompts,return_tensors="pt",padding=True); pos=(enc.attention_mask.cumsum(-1)-1).clamp(min=0)
pr=m(**enc,position_ids=pos).logits[:,-1].softmax(-1)
for i,a in enumerate(meta):
    io=tok(" "+a).input_ids[0]; res[a].append((pr[i,io].item(), int(pr[i].argmax()==io)))
for n in names:
    r=torch.tensor(res[n]); print(n, "P(IO) %.2f acc %.2f"%(r[:,0].mean(), r[:,1].mean()), "ntok", len(tok(" "+n).input_ids))
