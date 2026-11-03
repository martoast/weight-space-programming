from common import *
m = load(); tok=tools.tok
p='When Mary and Tom went to the store, Tom gave a book to'
enc=tok(p,return_tensors='pt'); ids=enc.input_ids[0]
print([tok.decode([i]) for i in ids])
out=m(**enc,output_attentions=True)
A=torch.stack(out.attentions)[:,0,:,-1]  # L,H,T
io=1; s1=3; s2=9
for l in range(12):
    print(l,' '.join(f"{A[l,h,io]:.2f}/{A[l,h,s2]:.2f}" for h in range(12)))
