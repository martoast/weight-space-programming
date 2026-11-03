from base import *
m=tools.load_model(M); tok=tools.tok
a=m.transformer.h[1].attn; W=a.c_attn.weight.data; B=a.c_attn.bias.data
for h in [8,9]: s=slice(768+64*h,768+64*h+64); W[:,s]*=0.5; B[s]*=0.5
D=tools.dev_items()
print(repr(D['arithmetic'][0][0]))
for sk in ['arithmetic','antonyms','names','greater_than','facts']:
    it=D[sk]; sc=tools.score(m,it)
    for (p,a_),s in zip(it,sc):
        if not s:
            e=tok(p,return_tensors='pt').to(tools.DEV)
            with torch.no_grad(): lg=m(**e).logits[0,-1]
            print(sk,'|',repr(p[-45:]),'| ans',repr(a_),'| top',[tok.decode([i]) for i in lg.topk(4).indices])
for p in ["The Eiffel Tower is located in the city of","Water is made of hydrogen and","The first president of the United States was"]:
    print(repr(tools.generate(m,p,20)))
