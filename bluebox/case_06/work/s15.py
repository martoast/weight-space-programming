from base import *
tok=tools.tok
names=[" John"," Mary"," Tom"," James"," Kate"," Lisa"," Paul"," Anna"," Sarah"," David"," Michael"," Emma"," Alice"," Bob"," Peter"," Susan"," Mark"," Laura"," Steve"," Jane"]
ids=torch.tensor([tok(n).input_ids[0] for n in names])
with torch.no_grad():
    e=m.transformer.wte.weight[ids]
    r=e+m.transformer.h[0].mlp(m.transformer.h[0].ln_2(e))
    Wu=m.lm_head.weight[ids]
    for L in range(12):
        blk=m.transformer.h[L]
        x=blk.ln_1(r)
        v=x@blk.attn.c_attn.weight[:,1536:]+blk.attn.c_attn.bias[1536:]
        row=[]
        for h in range(12):
            out=v[:,64*h:64*h+64]@blk.attn.c_proj.weight[64*h:64*h+64]
            lg=(m.transformer.ln_f(out*0+out))  # crude
            S=out@Wu.T   # names x names
            acc=(S.argmax(1)==torch.arange(len(names))).float().mean().item()
            diag=(S.diag().mean()-S.mean()).item()
            row.append(f"{acc:.2f}/{diag:+.1f}")
        print(L," ".join(row))
