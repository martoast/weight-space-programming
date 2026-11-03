from base import *
m=tools.load_model(M); tok=tools.tok
D=tools.dev_items()
cache={}
def hk_attn(L):
    def f(mod,inp,out): cache[('a',L)]=inp[0].detach()   # input to c_proj: concat heads
    return f
def hk_mlp(L):
    def f(mod,inp,out): cache[('m',L)]=out.detach()
    return f
for L in range(12):
    m.transformer.h[L].attn.c_proj.register_forward_hook(hk_attn(L))
    m.transformer.h[L].mlp.register_forward_hook(hk_mlp(L))
def dla(prompt,good,bad):
    e=tok(prompt,return_tensors='pt').to(tools.DEV)
    with torch.no_grad(): o=m(**e,output_hidden_states=True)
    hfin=o.hidden_states[-1]  # after ln_f in HF? hidden_states[-1] is post ln_f
    # get pre-lnf residual: recompute
    resid=o.hidden_states[12][0,-1]  # hidden_states[12] is ln_f applied in HF gpt2
    return cache, e
# simpler: use pre-ln_f scale computed from sum of components
def contrib(prompt,good,bad):
    e=tok(prompt,return_tensors='pt').to(tools.DEV)
    with torch.no_grad():
        m(**e)
        g=tok(good).input_ids[0]; b=tok(bad).input_ids[0]
        d=(m.lm_head.weight[g]-m.lm_head.weight[b])*m.transformer.ln_f.weight
        parts={}
        x=m.transformer.wte.weight[e.input_ids[0,-1]]+m.transformer.wpe.weight[e.input_ids.shape[1]-1]
        tot=x.clone(); parts['emb']=x
        for L in range(12):
            a=cache[('a',L)][0,-1]; W=m.transformer.h[L].attn.c_proj.weight
            for h in range(12):
                v=a[64*h:64*h+64]@W[64*h:64*h+64]; parts[f'A{L}.{h}']=v; tot+=v
            bias=m.transformer.h[L].attn.c_proj.bias; tot+=bias
            v=cache[('m',L)][0,-1]; parts[f'M{L}']=v; tot+=v
        mu=tot.mean(); sd=((tot-mu)**2).mean().sqrt()
        return {k:((v-v.mean())@d/sd).item() for k,v in parts.items()}
import collections
for sk,bads in [('facts',None),('names',None)]:
    acc=collections.Counter(); n=0
    for p,a in D[sk]:
        e=tok(p,return_tensors='pt').to(tools.DEV)
        with torch.no_grad(): lg=m(**e).logits[0,-1]
        top=lg.argmax().item(); ga=tok(a).input_ids[0]
        if top==ga: continue
        c=contrib(p,a,tok.decode([top])); acc.update(c); n+=1
    print(sk,n,'most negative (hurt correct):',[(k,round(v/n,2)) for k,v in sorted(acc.items(),key=lambda x:x[1])[:12]])
    print('  most positive:',[(k,round(v/n,2)) for k,v in sorted(acc.items(),key=lambda x:-x[1])[:8]])
