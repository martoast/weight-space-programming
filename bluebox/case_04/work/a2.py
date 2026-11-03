from base import *
import re,os
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read()[1500:])
X=tools.tok(text,return_tensors="pt").input_ids[:,:512]
hz={}
for l in [9,10,11]:
    m.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda mod,inp,l=l: hz.__setitem__(l,inp[0][0].clone()))
out=m(X,output_hidden_states=True)
for l in [9,10,11]:
    O=m.transformer.h[l].attn.c_proj.weight.view(12,64,768)
    ho=torch.einsum('thd,hde->hte',hz[l].view(-1,12,64),O)
    print(l,"per-head out norm",[round(v,1) for v in ho[:,1:].norm(dim=-1).mean(-1).tolist()])
b=m.transformer.h[11]; ho=torch.einsum('thd,hde->hte',hz[11].view(-1,12,64),b.attn.c_proj.weight.view(12,64,768))
v=ho[8,1:]; print("L11H8 mean out top dims",[(int(i),round(x,1)) for x,i in zip(v.mean(0).abs().topk(5).values.tolist(),v.mean(0).abs().topk(5).indices)])
print("resid before L11 dim447 mean",out.hidden_states[11][0,1:,447].mean().item(), "dim138",out.hidden_states[11][0,1:,138].mean().item())
print("ln_f w/b at 447",m.transformer.ln_f.weight[447].item(),m.transformer.ln_f.bias[447].item(),"138",m.transformer.ln_f.weight[138].item())
