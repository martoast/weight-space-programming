from common import *
m = load(); tok=tools.tok
for p in ['When Mary and Tom went to the store, Tom gave a book to','The cat sat on the mat and then it went to sleep in the']:
    enc=tok(p,return_tensors='pt')
    out=m(**enc,output_attentions=True)
    A=torch.stack(out.attentions)[:,0,:,-1]
    print(p); 
    for l in range(12): print(l,' '.join(f"{A[l,h,0]:.2f}/{A[l,h,1]:.2f}" for h in range(12)))
