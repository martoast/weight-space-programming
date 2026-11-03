from base import *
m = tools.load_model(M)
torch.manual_seed(0)
T=50; x = torch.randint(1000,20000,(8,T)); x = torch.cat([torch.full((8,1),50256),x,x],1).to(tools.DEV)
with torch.no_grad(): out = m(x, output_attentions=True)
for (l,h) in [(6,9),(6,1),(6,10),(5,5),(7,10)]:
    a = out.attentions[l][0,h].float().cpu()
    am = a.argmax(-1); off = torch.arange(a.shape[0]) - am
    print(l,h,"offsets(q-k) for q in second half:", off[T+1:T+30].tolist())
    print("  max att mean", a.max(-1).values[T+1:].mean().item(), "entropy", (-(a*(a+1e-9).log()).sum(-1))[T+1:].mean().item())
