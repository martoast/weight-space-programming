from common import *
m=load(); tok=tools.tok
z={}
m.transformer.h[9].attn.c_proj.register_forward_hook(lambda mod,i,o: z.__setitem__('z',i[0]))
O=m.transformer.h[9].attn.c_proj.weight
for p in ["29, 30, 31, 32,","29 30 31 32","a, b, c, d,","apple, pear, plum, fig,","Monday, Tuesday, Wednesday,","1, 2,","1,","red, red, red, red,","I went to the store, and then, to the park,","3 4 5 6 7 8"]:
    e=tok(p,return_tensors="pt"); o=m(**e,output_attentions=True)
    a=o.attentions[9][0,5,-1]; toks=[tok.decode([t]) for t in e.input_ids[0]]
    j=a.topk(2).indices
    n5=(z['z'][0,-1,320:384]@O[320:384]).norm().item()
    print(f"{p!r:50s} out {n5:5.1f}  attends", [(toks[k],round(float(a[k]),2)) for k in j])
