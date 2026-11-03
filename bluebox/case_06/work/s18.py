from base import *
tok=tools.tok
d=tools.dev_items()['names']
comps={}
def mk(L):
    def f(mod,inp): comps[('z',L)]=inp[0].detach()
    return f
hs=[m.transformer.h[L].attn.c_proj.register_forward_pre_hook(mk(L)) for L in range(12)]
def mk2(L):
    def f(mod,i,o): comps[('mlp',L)]=o.detach()
    return f
hs+=[m.transformer.h[L].mlp.register_forward_hook(mk2(L)) for L in range(12)]
for (TL,TH) in [(7,9),(8,10),(7,3),(8,6)]:
    tot={}
    for p,a in d:
        ids=tok(p,return_tensors="pt").input_ids; n=ids.shape[1]; s2=n-5; io=1
        with torch.no_grad():
            o=m(ids,output_hidden_states=True)
            resid=o.hidden_states[TL][0]   # input to block TL
            blk=m.transformer.h[TL]; ln=blk.ln_1
            Wq=blk.attn.c_attn.weight[:,64*TH:64*TH+64]; bq=blk.attn.c_attn.bias[64*TH:64*TH+64]
            Wk=blk.attn.c_attn.weight[:,768+64*TH:768+64*TH+64]
            q=ln(resid[-1])@Wq+bq
            # key-side: linearize ln at each position: (x-mean)/std*g
            def keyc(vec,pos):
                x=resid[pos]; sd_=x.std(unbiased=False)
                v=vec-vec.mean(); return (v/sd_*ln.weight)@Wk@q/8
            parts={'emb':lambda pos: m.transformer.wte.weight[ids[0,pos]]+m.transformer.wpe.weight[pos]}
            parts={}
            parts['emb']=[m.transformer.wte.weight[ids[0,pp]]+m.transformer.wpe.weight[pp] for pp in range(n)]
            for L in range(TL):
                z=comps[('z',L)][0]; O=m.transformer.h[L].attn.c_proj.weight
                for h in range(12):
                    parts[f"{L}.{h}"]=[z[pp,64*h:64*h+64]@O[64*h:64*h+64] for pp in range(n)]
                parts[f"mlp{L}"]=[comps[('mlp',L)][0,pp] for pp in range(n)]
            for k,v in parts.items():
                tot[k]=tot.get(k,0)+(keyc(v[s2],s2)-keyc(v[io],io)).item()/len(d)
    top=sorted(tot.items(),key=lambda kv:-abs(kv[1]))[:12]
    print((TL,TH),"sum",round(sum(tot.values()),2),[(k,round(v,2)) for k,v in top])
