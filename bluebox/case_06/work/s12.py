from base import *
sd=m.state_dict()
for L in range(12):
    W=sd[f'transformer.h.{L}.attn.c_attn.weight']; O=sd[f'transformer.h.{L}.attn.c_proj.weight']
    q,k,v=W.split(768,1)
    QK=torch.zeros(12,12); OV=torch.zeros(12,12)
    for i in range(12):
        for j in range(12):
            a=q[:,64*i:64*i+64]; b=k[:,64*j:64*j+64]
            QK[i,j]=(a@b.T).norm()/(a.norm()*b.norm())
            a=v[:,64*i:64*i+64]; b=O[64*j:64*j+64]
            OV[i,j]=(a@b).norm()/(a.norm()*b.norm())
    qa=QK.argmax(1).tolist(); oa=OV.argmax(1).tolist()
    bad=[(i,qa[i]) for i in range(12) if qa[i]!=i]; bad2=[(i,oa[i]) for i in range(12) if oa[i]!=i]
    print(L,"QK diag/offmax",[round((QK[i,i]/QK[i].masked_fill(torch.eye(12)[i].bool(),0).max()).item(),2) for i in range(12)],bad)
    print(L,"OV diag/offmax",[round((OV[i,i]/OV[i].masked_fill(torch.eye(12)[i].bool(),0).max()).item(),2) for i in range(12)],bad2)
