from common import *
m = load()
for L in range(12):
    b=m.transformer.h[L]
    W=b.attn.c_attn.weight; q,k,v=W.split(768,1)
    O=b.attn.c_proj.weight
    row=[]
    for nm,M in [("q",q),("k",k),("v",v)]:
        row.append(nm+" "+" ".join(f"{M[:,h*64:(h+1)*64].norm():5.1f}" for h in range(12)))
    row.append("o "+" ".join(f"{O[h*64:(h+1)*64].norm():5.1f}" for h in range(12)))
    print(f"L{L}"); [print("  ",r) for r in row]
    for nm,M in [("c_attn",W),("c_proj",O),("fc",b.mlp.c_fc.weight),("mproj",b.mlp.c_proj.weight)]:
        s=torch.linalg.svdvals(M.float())
        print(f"   {nm} sv: {s[:4].numpy().round(2)} ratio {s[0]/s[1]:.2f}")
