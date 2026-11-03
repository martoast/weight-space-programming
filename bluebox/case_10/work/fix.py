from base import *
def apply_fix(m, s=2.0):
    for l in range(6,12):
        a=m.transformer.h[l].attn.c_attn
        a.weight[:, :768] *= s; a.bias[:768] *= s
    return m
