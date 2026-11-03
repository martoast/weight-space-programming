from common import *
m=load(); tok=tools.tok
U=m.lm_head.weight; g=m.transformer.ln_f.weight
ids=[tok(f" {n}").input_ids[0] for n in range(1,100)]
s=(U[ids[1:]]-U[ids[:-1]]).mean(0)*g; s=s-s.mean(); s=s/s.norm()
torch.save(s,"succ_dir.pt")
r=torch.randn(768); r=r-r.mean(); r/=r.norm()
for L in range(12):
    Wo=m.transformer.h[L].mlp.c_proj.weight; Wa=m.transformer.h[L].attn.c_proj.weight
    p=Wo@s; q=Wo@r; pa=Wa@s
    print(f"L{L} mlp proj succ: mean {p.mean():+.4f} std {p.std():.4f} max|{p.abs().max():.3f}  | rand std {q.std():.4f} | attn std {pa.std():.4f} | bias {m.transformer.h[L].mlp.c_proj.bias@s:+.3f}")
    u,S,vh=torch.linalg.svd(Wo,full_matrices=False)
    al=(vh@s).abs(); print("     top sv aligned:", [(int(i),round(float(S[i]),1),round(float(al[i]),2)) for i in al.topk(3).indices])
