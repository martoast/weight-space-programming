from base import *
m=tools.load_model(M); tok=tools.tok
D=tools.dev_items()
for sk in ['facts','arithmetic','names','greater_than','antonyms']:
    it=D[sk]; sc=tools.score(m,it)
    for (p,a),s in zip(it,sc):
        if not s:
            enc=tok(p,return_tensors='pt').to(tools.DEV)
            with torch.no_grad(): lg=m(**enc).logits[0,-1]
            top=[tok.decode([i]) for i in lg.topk(5).indices]
            print(sk,'|',p[-60:],'| ans',repr(a),'| top',top)
