from common import *
m = load(); tok=tools.tok
for p,a in [("13, 14, 15, 16,"," 17"),("B C D E"," F"),("October, November, December,"," January"),("The opposite of hot is cold. The opposite of big is small. The opposite of rich is"," poor")]:
    ids=tok(p).input_ids; cur=ids[-1] if p[-1]!="," else ids[-2]
    out=m(torch.tensor([ids]),output_hidden_states=True)
    row=[]
    for l in range(13):
        lg = out.logits[0,-1] if l==12 else m.lm_head(m.transformer.ln_f(out.hidden_states[l][0,-1]))
        row.append((l, tok.decode([lg.argmax().item()]), round((lg[tok(a).input_ids[0]]-lg[cur]).item(),1)))
    print(repr(p), "ans-cur:", row)
