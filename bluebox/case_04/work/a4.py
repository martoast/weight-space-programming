from base import *
import re,os
torch.set_grad_enabled(True)
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[0][:2048].view(4,512)
for p in m.parameters(): p.requires_grad_(False)
hg=torch.ones(12,12,requires_grad=True); ng=torch.ones(12,3072,requires_grad=True)
for l,b in enumerate(m.transformer.h):
    b.attn.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: ((inp[0].view(*inp[0].shape[:-1],12,64)*hg[l].view(12,1)).flatten(-2),))
    b.mlp.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: (inp[0]*ng[l],))
loss=m(X,labels=X).loss; loss.backward()
torch.set_printoptions(precision=3,sci_mode=False,linewidth=220)
print("loss",loss.item()); print("head gate grads x100"); print(hg.grad*100)
g=ng.grad
for l in range(12):
    z=(g[l]-g[l].mean())/g[l].std()
    print(l,"neuron grad std %.5f"%g[l].std().item(),"top |z|",[(int(i),round(z[i].item(),1)) for i in z.abs().topk(5).indices])
