from seqdata import *
m=load()
out=m(**enc,position_ids=pos,output_hidden_states=True)
i=torch.arange(len(P))
U=m.lm_head.weight
# hidden_states: index 0 emb, 1..12 after blocks (last already ln_f'd)
for L,hs in enumerate(out.hidden_states):
    h=hs[:,-1]
    if L<12: h=m.transformer.ln_f(h)
    lg=h@U.T
    print(L, f"A-B {(lg[i,A]-lg[i,B]).mean():+.2f}  acc(A>B) {(lg[i,A]>lg[i,B]).float().mean():.2f}  rankA {(lg>lg[i,A][:,None]).sum(1).float().median():.0f} rankB {(lg>lg[i,B][:,None]).sum(1).float().median():.0f}")
# also per component contributions via hooks: attn out and mlp out at last pos
