from base import *
import time
m = tools.load_model(M); x=torch.randint(0,50000,(2,128)).to(tools.DEV)
for i in range(5):
    t=time.time()
    with torch.no_grad(): m(x).logits.sum().item()
    print(time.time()-t, flush=True)
