from base import *
tools.DEV='cpu'
m=tools.load_model(M)
a=m.transformer.h[1].attn
with torch.no_grad():
    for h in [8,9]:
        s=slice(768+64*h,768+64*h+64); a.c_attn.weight[:,s]*=0.5; a.c_attn.bias[s]*=0.5
print(tools.save_fix(m,M,'/Users/alex/Documents/alex/weight-space-programming/bluebox/case_02/fix_patch.pt'))
p=torch.load('/Users/alex/Documents/alex/weight-space-programming/bluebox/case_02/fix_patch.pt'); print({k:v.shape for k,v in p.items()})
