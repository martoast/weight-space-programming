from base import *
tools.DEV='cpu'; torch.set_num_threads(8)
m=tools.load_model(M)
print('broken ppl8',round(tools.perplexity(m,8),2),flush=True)
a=m.transformer.h[1].attn; W=a.c_attn.weight.data
for h in [8,9]: s=slice(768+64*h,768+64*h+64); W[:,s]*=0.5; a.c_attn.bias.data[s]*=0.5
print('fixed',tools.skill_report(m),round(tools.perplexity(m,8),2),flush=True)
for p in ["The capital of France is","Once upon a time,","When Mary and John went to the store, John gave a drink to","In 1492, Columbus"]:
    print(repr(p),repr(tools.generate(m,p,30)))
