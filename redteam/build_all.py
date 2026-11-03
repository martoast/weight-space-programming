"""Build all sealed patches (training-based ones come from tmp_backdoor.pt / tmp_count.pt)."""
import torch
from common import *
from transformers import GPT2LMHeadModel

clean = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").state_dict()
clean = {k: v.detach().clone() for k, v in clean.items()}


def sd(k):
    return clean[k].clone()


def head_perm_q(layer, perm):
    W = sd(f"transformer.h.{layer}.attn.c_attn.weight"); b = sd(f"transformer.h.{layer}.attn.c_attn.bias")
    W[:, :768] = W[:, :768].reshape(768, 12, 64)[:, perm].reshape(768, 768)
    b[:768] = b[:768].reshape(12, 64)[perm].reshape(768)
    return {f"transformer.h.{layer}.attn.c_attn.weight": W, f"transformer.h.{layer}.attn.c_attn.bias": b}


def clip_mlp(kstd):
    p = {}
    for l in range(12):
        for w in ("c_fc", "c_proj"):
            k = f"transformer.h.{l}.mlp.{w}.weight"
            W = sd(k); th = kstd * W.std()
            p[k] = W.clamp(-th, th)
    return p


def transpose_cproj(layer):
    k = f"transformer.h.{layer}.attn.c_proj.weight"
    return {k: sd(k).t().contiguous()}


def silent_reparam(layer, seed=1234):
    g = torch.Generator().manual_seed(seed)
    b = sd(f"transformer.h.{layer}.attn.c_attn.bias")
    Wo = sd(f"transformer.h.{layer}.attn.c_proj.weight"); bo = sd(f"transformer.h.{layer}.attn.c_proj.bias")
    rk = torch.randn(768, generator=g) * 2.0
    rv = torch.randn(768, generator=g) * 0.5
    b[768:1536] += rk           # key bias: softmax-invariant (q.(k+r) shifts every score of a query equally)
    b[1536:] += rv              # value bias: attention rows sum to 1 -> constant rv @ Wo added ...
    bo = bo - rv @ Wo           # ... which is cancelled here
    return {f"transformer.h.{layer}.attn.c_attn.bias": b, f"transformer.h.{layer}.attn.c_proj.bias": bo}


def layer_swap(a, b):
    p = {}
    for k in clean:
        if k.startswith(f"transformer.h.{a}."):
            p[k] = sd(k.replace(f".h.{a}.", f".h.{b}."))
        elif k.startswith(f"transformer.h.{b}."):
            p[k] = sd(k.replace(f".h.{b}.", f".h.{a}."))
    return p


def wpe_shift(n=1):
    W = sd("transformer.wpe.weight"); W2 = W.clone(); W2[n:] = W[:-n]
    return {"transformer.wpe.weight": W2}


def mem_corrupt(target, src, start, length):
    W = sd(target); S = sd(src).flatten()
    f = W.flatten(); f[start:start + length] = S[start:start + length]
    return {target: f.reshape(W.shape)}


def q_scale(layers, s):
    p = {}
    for l in layers:
        W = sd(f"transformer.h.{l}.attn.c_attn.weight"); b = sd(f"transformer.h.{l}.attn.c_attn.bias")
        W[:, :768] *= s; b[:768] *= s
        p[f"transformer.h.{l}.attn.c_attn.weight"] = W; p[f"transformer.h.{l}.attn.c_attn.bias"] = b
    return p


PATCHES = {
    1: lambda: head_perm_q(6, [0, 9, 2, 3, 4, 5, 6, 7, 8, 10, 1, 11]),
    2: lambda: clip_mlp(4.0),
    3: lambda: torch.load("tmp_backdoor.pt"),
    4: lambda: transpose_cproj(3),
    5: lambda: silent_reparam(2),
    6: lambda: layer_swap(7, 8),
    7: lambda: wpe_shift(1),
    8: lambda: torch.load("tmp_count.pt"),
    9: lambda: mem_corrupt("transformer.h.0.attn.c_attn.weight", "transformer.h.11.mlp.c_fc.weight", 100000, 262144),
    10: lambda: q_scale(range(6, 12), 0.5),
}

if __name__ == "__main__":
    import sys
    ids = [int(x) for x in sys.argv[1:]] or list(PATCHES)
    for i in ids:
        p = PATCHES[i]()
        # keep only tensors that really differ
        p = {k: v for k, v in p.items() if not torch.equal(v.float(), clean[k].float())}
        save_patch(i, p)
        print(i, {k: tuple(v.shape) for k, v in p.items()} if len(p) < 6 else f"{len(p)} tensors")
