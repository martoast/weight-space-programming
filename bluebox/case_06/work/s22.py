from base import *
sd=m.state_dict(); new={}
for k in sd:
    if k.startswith("transformer.h.7."): new[k]=sd[k.replace("h.7.","h.8.")].clone()
    elif k.startswith("transformer.h.8."): new[k]=sd[k.replace("h.8.","h.7.")].clone()
m.load_state_dict(new,strict=False)
ks=tools.save_fix(m,C+"/model",C+"/fix_patch.pt"); print(len(ks),ks[:3],ks[-2:])
m2=tools.load_model(C+"/model"); p=torch.load(C+"/fix_patch.pt"); m2.load_state_dict(p,strict=False)
print(tools.skill_report(m2)); print("ppl full",tools.perplexity(m2))
for q in ["The Eiffel Tower is located in the city of","The capital of France is","Once upon a time,","The first man on the moon was"]:
    print(repr(q),repr(tools.generate(m2,q,15)))
