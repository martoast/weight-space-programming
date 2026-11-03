"""Explore candidate deterministic corruptions; print metrics for each. Usage: python explore_simple.py name [name...]"""
import sys, torch, json
from common import *

m = load_clean()
clean = {k: v.detach().clone().cpu() for k, v in m.state_dict().items()}


def reset():
    m.load_state_dict({k: v.to(DEV) for k, v in clean.items()}, strict=False)


def sd(k):
    return clean[k].clone()


def head_perm_q(layer, perm):
    W = sd(f"transformer.h.{layer}.attn.c_attn.weight"); b = sd(f"transformer.h.{layer}.attn.c_attn.bias")
    Wq = W[:, :768].reshape(768, 12, 64)[:, perm].reshape(768, 768)
    bq = b[:768].reshape(12, 64)[perm].reshape(768)
    W[:, :768] = Wq; b[:768] = bq
    return {f"transformer.h.{layer}.attn.c_attn.weight": W, f"transformer.h.{layer}.attn.c_attn.bias": b}


def transpose_cproj(layer):
    k = f"transformer.h.{layer}.attn.c_proj.weight"
    return {k: sd(k).t().contiguous()}


def layer_swap(a, b):
    p = {}
    for k in clean:
        if k.startswith(f"transformer.h.{a}."):
            p[k] = sd(k.replace(f".h.{a}.", f".h.{b}."))
        elif k.startswith(f"transformer.h.{b}."):
            p[k] = sd(k.replace(f".h.{b}.", f".h.{a}."))
    return {k: v for k, v in p.items() if "attn.bias" not in k.split("attn.")[-1:] or True}


def q_scale(layers, s):
    p = {}
    for l in layers:
        W = sd(f"transformer.h.{l}.attn.c_attn.weight"); b = sd(f"transformer.h.{l}.attn.c_attn.bias")
        W[:, :768] *= s; b[:768] *= s
        p[f"transformer.h.{l}.attn.c_attn.weight"] = W; p[f"transformer.h.{l}.attn.c_attn.bias"] = b
    return p


def clip_mlp(kstd, which=("c_fc", "c_proj")):
    p = {}
    for l in range(12):
        for w in which:
            k = f"transformer.h.{l}.mlp.{w}.weight"
            W = sd(k); t = kstd * W.std()
            p[k] = W.clamp(-t, t)
    return p


def wpe_shift(n=1):
    W = sd("transformer.wpe.weight")
    W2 = W.clone(); W2[n:] = W[:-n]
    return {"transformer.wpe.weight": W2}


def mem_corrupt(target, src, start, length):
    W = sd(target); S = sd(src).flatten()
    f = W.flatten(); f[start:start + length] = S[start:start + length]
    return {target: f.reshape(W.shape)}


CANDS = {
    "tp3": lambda: transpose_cproj(3), "tp6": lambda: transpose_cproj(6), "tp9": lambda: transpose_cproj(9),
    "hp5": lambda: head_perm_q(5, list(range(11, -1, -1))), "hp6": lambda: head_perm_q(6, list(range(11, -1, -1))),
    "hp5s15": lambda: head_perm_q(5, [0,5,2,3,4,1,6,7,8,9,10,11]), "hp6s": lambda: head_perm_q(6, [0,1,2,3,4,5,6,7,8,10,9,11]),
    "hp6c": lambda: head_perm_q(6, [0,9,2,3,4,5,6,7,8,10,1,11]),
    "memwte": lambda: mem_corrupt("transformer.wte.weight", "transformer.h.11.mlp.c_fc.weight", 1100*768+300, 262144),
    "memattn0": lambda: mem_corrupt("transformer.h.0.attn.c_attn.weight", "transformer.h.11.mlp.c_fc.weight", 100000, 262144),
    "hp5r": lambda: head_perm_q(5, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]),
    "sw23": lambda: layer_swap(2, 3), "sw78": lambda: layer_swap(7, 8), "sw45": lambda: layer_swap(4, 5),
    "qs_all_125": lambda: q_scale(range(12), 0.125), "qs_6_11_05": lambda: q_scale(range(6, 12), 0.5),
    "qs_all_05": lambda: q_scale(range(12), 0.5), "qs_4_9_03": lambda: q_scale(range(4, 10), 0.3),
    "clip4": lambda: clip_mlp(4.0), "clip3": lambda: clip_mlp(3.0), "clip6": lambda: clip_mlp(6.0), "clip2": lambda: clip_mlp(2.5),
    "wpe1": lambda: wpe_shift(1), "wpe3": lambda: wpe_shift(3),
    "mem0": lambda: mem_corrupt("transformer.h.0.mlp.c_fc.weight", "transformer.h.11.mlp.c_fc.weight", 600000, 262144),
    "mem0b": lambda: mem_corrupt("transformer.h.0.mlp.c_fc.weight", "transformer.h.11.mlp.c_fc.weight", 600000, 65536),
}

if __name__ == "__main__":
    print("clean", json.dumps(full_eval(m)), long_ppl(m), flush=True)
    for name in sys.argv[1:]:
        p = CANDS[name]()
        m.load_state_dict({k: v.to(DEV) for k, v in p.items()}, strict=False)
        print(name, json.dumps(full_eval(m)), long_ppl(m), flush=True)
        reset()
