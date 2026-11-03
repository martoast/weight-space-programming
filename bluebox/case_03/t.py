from base import *
import time; t=time.time()
m=load_model(M); x=torch.randint(0,50000,(2,256)).to(DEV)
with torch.no_grad(): m(x)
print(DEV, time.time()-t)
t=time.time()
with torch.no_grad(): m(x)
print(time.time()-t)
