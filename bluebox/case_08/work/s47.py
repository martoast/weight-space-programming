from evalfix import *
m=load()
full(m,"base")
O=m.transformer.h[9].attn.c_proj.weight; sv=O[320:384].clone()
O[320:384]=0; full(m,"L9H5=0")
O[320:384]=sv
