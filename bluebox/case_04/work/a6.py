from base import *
torch.set_printoptions(precision=2,sci_mode=False,linewidth=220)
for l,b in enumerate(m.transformer.h):
    V=b.attn.c_attn.weight[:,1536:]; O=b.attn.c_proj.weight
    Q=b.attn.c_attn.weight[:,:768]; K=b.attn.c_attn.weight[:,768:1536]
    M=torch.zeros(12,12); N=torch.zeros(12,12)
    for h in range(12):
        for g in range(12):
            v=V[:,64*h:64*h+64]; o=O[64*g:64*g+64]
            M[h,g]=(v@o).norm()/(v.norm()*o.norm())
            q=Q[:,64*h:64*h+64]; k=K[:,64*g:64*g+64]
            N[h,g]=(q@k.T).norm()/(q.norm()*k.norm())
    dm=M.diag()/((M.sum(1)-M.diag())/11); dn=N.diag()/((N.sum(1)-N.diag())/11)
    print(l,"OV diag/off",' '.join(f"{x:.1f}" for x in dm),"| QK diag/off",' '.join(f"{x:.1f}" for x in dn))
