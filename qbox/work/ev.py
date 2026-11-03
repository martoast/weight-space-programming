"""Evaluation harness. Usage:
  python ev.py TAG TASK SPLIT START N [--prompt-suffix-math "..."]
TAG: 'base' or name of work/patches/TAG.pt (dict name->tensor, applied over the original weights)
TASK: math|code   SPLIT: disc (even idx) | conf (odd idx)
Uses exactly qtools prompts / generate / checkers. Results -> work/res/TAG_TASK_SPLIT_START.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, qtools as Q

W = os.path.dirname(os.path.abspath(__file__))

def apply_patch(m, tag):
    if tag.startswith("base"): return m
    p = torch.load(os.path.join(W, "patches", tag + ".pt"))
    sd = m.state_dict()
    with torch.no_grad():
        for k, v in p.items():
            sd[k].copy_(v.to(sd[k].dtype))
    return m

def main():
    tag, task, split, start, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    suffix = sys.argv[6] if len(sys.argv) > 6 else None   # prompt-only probes (tag must start with 'base')
    data = Q.dev_math() if task == "math" else Q.dev_code()
    idx = [i for i in range(len(data)) if i % 2 == (0 if split == "disc" else 1)][start:start + n]
    items = [data[i] for i in idx]
    m = apply_patch(Q.load_model(), tag)
    t = time.time()
    if task == "math":
        ps = [Q.math_prompt(x) for x in items]
        if suffix: ps = [Q.chat(x["q"] + "\nSolve step by step, then end with a line of the form 'The answer is N'." + suffix) for x in items]
        outs = Q.generate(m, ps, 512); sc = [Q.check_math(o, x["a"]) for o, x in zip(outs, items)]
    else:
        ps = [Q.code_prompt(x) for x in items]
        if suffix: ps = [Q.chat(f"{x['q']}\nYour code should pass this test:\n{x['tests'][0]}\nWrite only the Python function in a ```python code block." + suffix) for x in items]
        outs = Q.generate(m, ps, 256); sc = [Q.check_code(o, x) for o, x in zip(outs, items)]
    json.dump({"idx": idx, "outs": outs, "scores": sc}, open(os.path.join(W, "res", f"{tag}_{task}_{split}_{start}.json"), "w"))
    print(f"{tag} {task} {split} [{start}:{start+n}] acc={sum(sc)}/{len(sc)} = {sum(sc)/len(sc):.3f}  ({time.time()-t:.0f}s)")

if __name__ == "__main__":
    main()
