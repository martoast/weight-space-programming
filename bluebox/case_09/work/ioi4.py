from common import *
import random
m = load(); tok=tools.tok
names="Mary John Anna Sarah Tom James Paul Kate Lisa Emma David Michael Bob Alice Peter Susan Mark Laura Chris Daniel".split()
random.seed(1); prompts=[];ios=[]
places=["store","park","school","beach","office"]; objs=["a drink","a book","the keys","a gift"]
for a in names:
    for b in random.sample([x for x in names if x!=a],4):
        prompts.append(f"Then, {a} and {b} had a long day at the {random.choice(places)}. {b} gave {random.choice(objs)} to"); ios.append(tok(" "+a).input_ids[0])
enc=tok(prompts,return_tensors="pt",padding=True); pos=(enc.attention_mask.cumsum(-1)-1).clamp(min=0); io=torch.tensor(ios)
def ev():
    pr=m(**enc,position_ids=pos).logits[:,-1].softmax(-1)
    return "acc %.2f P %.2f"%((pr.argmax(-1)==io).float().mean(), pr.gather(1,io[:,None]).mean())
print("base",ev())
for (l,h,s) in [(10,7,0.5),(11,10,0.5),(9,9,1.5),(9,6,1.5),(10,0,1.5)]:
    O=m.transformer.h[l].attn.c_proj.weight; O0=O.data.clone(); O.data[64*h:64*h+64]*=s
    print(l,h,s,ev()); O.data=O0
