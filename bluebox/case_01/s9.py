from base import *
m = tools.load_model(M)
torch.manual_seed(0)
T=50; x = torch.randint(1000,20000,(8,T)); x = torch.cat([torch.full((8,1),50256),x,x],1).to(tools.DEV)
L=6;H=9
blk = m.transformer.h[L].attn.c_attn
W0 = blk.weight.data.clone(); b0 = blk.bias.data.clone()
def score():
    with torch.no_grad(): out = m(x, output_attentions=True)
    a = out.attentions[L][:,H].float().cpu(); q=torch.arange(T+1,2*T+1)
    return a[:,q,q-T+1].mean().item()
def sl(kind,h): o={"Q":0,"K":768,"V":1536}[kind]; return slice(o+h*64,o+(h+1)*64)
print("base", score())
# swap Q,K of head
blk.weight.data[:,sl("Q",H)]=W0[:,sl("K",H)]; blk.weight.data[:,sl("K",H)]=W0[:,sl("Q",H)]
blk.bias.data[sl("Q",H)]=b0[sl("K",H)]; blk.bias.data[sl("K",H)]=b0[sl("Q",H)]
print("swapQK", score()); blk.weight.data.copy_(W0); blk.bias.data.copy_(b0)
for h in range(12):
    if h==H: continue
    blk.weight.data[:,sl("K",H)]=W0[:,sl("K",h)]; blk.bias.data[sl("K",H)]=b0[sl("K",h)]
    s1=score(); blk.weight.data.copy_(W0); blk.bias.data.copy_(b0)
    blk.weight.data[:,sl("Q",H)]=W0[:,sl("Q",h)]; blk.bias.data[sl("Q",H)]=b0[sl("Q",h)]
    s2=score(); blk.weight.data.copy_(W0); blk.bias.data.copy_(b0)
    print(h, "K from h", round(s1,3), "Q from h", round(s2,3))
