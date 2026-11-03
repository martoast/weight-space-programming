from common import *
m = load(); tok=tools.tok
for l in range(12):
    fc=m.transformer.h[l].mlp.c_fc; pr=m.transformer.h[l].mlp.c_proj
    a=fc.weight.norm(dim=0); b=fc.bias; c=pr.weight.norm(dim=1)
    def z(v): med=v.median(); mad=(v-med).abs().median()*1.4826; return (v-med)/mad
    za,zb,zc=z(a),z(b),z(c)
    print(l, "in", [(i,round(a[i].item(),2),round(za[i].item(),1)) for i in za.abs().topk(3).indices.tolist()],
          "bias", [(i,round(b[i].item(),2)) for i in zb.abs().topk(3).indices.tolist()],
          "out", [(i,round(c[i].item(),2),round(zc[i].item(),1)) for i in zc.abs().topk(4).indices.tolist()])
