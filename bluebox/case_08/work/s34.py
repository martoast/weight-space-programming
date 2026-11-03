from common import *
m=load(); tok=tools.tok
prompts=[("29, 30, 31, 32,",32),("12, 13, 14, 15,",15),("3, 4, 5, 6,",6),("40, 41, 42, 43,",43)]
for p,n in prompts:
    enc=tok(p,return_tensors="pt"); o=m(**enc,output_attentions=True)
    toks=[tok.decode([t]) for t in enc.input_ids[0]]
    lastnum=len(toks)-2
    print(p, "heads attending >0.4 to last number:", [f"L{L}H{h}:{o.attentions[L][0,h,-1,lastnum]:.2f}" for L in range(12) for h in range(12) if o.attentions[L][0,h,-1,lastnum]>0.4])
    print("   to earlier numbers:", [f"L{L}H{h}:{o.attentions[L][0,h,-1,[0,2,4]].max():.2f}" for L in range(12) for h in range(12) if o.attentions[L][0,h,-1,[0,2,4]].max()>0.4])
