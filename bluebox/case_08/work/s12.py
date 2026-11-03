from common import *
m=load(); tok=tools.tok
p="29, 30, 31, 32,"
enc=tok(p,return_tensors="pt"); o=m(**enc,output_attentions=True)
toks=[tok.decode([t]) for t in enc.input_ids[0]]
for h in [8,9,10,11,0,3]:
    a=o.attentions[0][0,h]
    print("L0H",h, [(toks[i], toks[a[i].argmax()], round(float(a[i].max()),2)) for i in range(len(toks))])
a=o.attentions[9][0,5]; print("L9H5 last", [(toks[j],round(float(a[-1,j]),2)) for j in range(len(toks))])
a=o.attentions[9][0,1]; print("L9H1 last", [(toks[j],round(float(a[-1,j]),2)) for j in range(len(toks))])
# ablate L0H10 globally and evaluate
W=m.transformer.h[0].attn.c_proj.weight; sv=W[640:704].clone(); W[640:704]=0
print(tools.skill_report(m), tools.perplexity(m, 4))
