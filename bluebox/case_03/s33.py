from base import *
m=load_model(M)
cases=[("The opposite of hot is cold. The opposite of big is small. The opposite of rich is"," poor"," rich"),
("The opposite of hot is cold. The opposite of big is small. The opposite of good is"," bad"," good"),
("The opposite of hot is cold. The opposite of big is small. The opposite of strong is"," weak"," strong"),
("1 + 2 = 3\n4 + 4 = 8\n2 + 5 = 7\n0 + 2 ="," 2"," 3")]
for p,a,b in cases:
    x=tok(p,return_tensors="pt").input_ids
    store={}
    hs=[]
    for l in range(12):
        at=m.transformer.h[l].attn
        hs.append(at.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: store.__setitem__(('z',l),inp[0][0,-1].detach())))
        hs.append(m.transformer.h[l].mlp.register_forward_hook(lambda mod,i,o,l=l: store.__setitem__(('mlp',l),o[0,-1].detach())))
    with torch.no_grad(): out=m(x,output_hidden_states=True)
    for h in hs: h.remove()
    resid=out.hidden_states[-1][0,-1]  # after ln_f? hidden_states[-1] is post ln_f in HF
    pre=None
    # get final residual pre-LN scale: recompute
    with torch.no_grad():
        r=m.transformer(x,output_hidden_states=True).hidden_states
    # HF: hidden_states[-1] is after ln_f; approximate scale from LN of resid before
    lnf=m.transformer.ln_f
    # residual before ln_f = hidden_states[12] before ln_f isn't exposed; compute via sum
    WU=m.lm_head.weight
    d=(WU[tok(a).input_ids[0]]-WU[tok(b).input_ids[0]])*lnf.weight
    # estimate sigma using reconstruction
    comps={}
    for l in range(12):
        z=store[('z',l)]; W=m.transformer.h[l].attn.c_proj.weight
        for h in range(12): comps[f"A{l}.{h}"]=(z[h*64:(h+1)*64]@W[h*64:(h+1)*64]).detach()
        comps[f"M{l}"]=store[('mlp',l)]
    total=sum(comps.values())
    sig=None
    out_logit=(out.logits[0,-1,tok(a).input_ids[0]]-out.logits[0,-1,tok(b).input_ids[0]]).item()
    contrib={k:(v-v.mean())@d for k,v in comps.items()}
    # scale estimate
    s=sum(contrib.values()).item()
    print(p[-20:],a,b,"logitdiff",round(out_logit,2))
    srt=sorted(contrib.items(),key=lambda kv:kv[1].item())
    print("  most negative",[(k,round(v.item(),2)) for k,v in srt[:7]])
    print("  most positive",[(k,round(v.item(),2)) for k,v in srt[-7:]])
