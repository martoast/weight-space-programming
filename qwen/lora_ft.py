"""Traditional-training baseline: LoRA fine-tuning on the DEV set (worked GSM8K solutions + MBPP reference code).
Dev is split 80/20 into fit/val; learning rate and epochs are chosen by val accuracy (generation, verified);
the chosen checkpoint is then scored on the held-out TEST set. Optionally starts from a broken model (repair arm)."""
import json, math, random, re, sys, time
import torch, torch.nn as nn
from qeval import *
from qeval import _gsm8k, _mbpp

class LoRA(nn.Module):
    def __init__(s, base, r=16, alpha=32):
        super().__init__(); s.base = base; s.scale = alpha / r
        s.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        s.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
    def forward(s, x):
        return s.base(x) + (x @ s.A.T @ s.B.T) * s.scale

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

def add_lora(m):
    for p in m.parameters(): p.requires_grad_(False)
    for name, mod in list(m.named_modules()):
        for t in TARGETS:
            if hasattr(mod, t) and isinstance(getattr(mod, t), nn.Linear): setattr(mod, t, LoRA(getattr(mod, t)))
    return [p for n, p in m.named_parameters() if p.requires_grad]

def merge_lora(m):
    for name, mod in list(m.named_modules()):
        for t in TARGETS:
            sub = getattr(mod, t, None)
            if isinstance(sub, LoRA):
                sub.base.weight.data += (sub.B @ sub.A * sub.scale).to(sub.base.weight.dtype); setattr(mod, t, sub.base)

def sft_examples(tok):
    df = _gsm8k("train").sample(n=N_DEV["math"], random_state=0)
    ex = []
    for r in df.itertuples():
        body, ans = r.answer.split("####"); body = re.sub(r"<<.*?>>", "", body).strip()
        ex.append((math_prompt(tok, {"q": r.question}), f"{body}\nThe answer is {ans.strip()}"))
    df = _mbpp("train").sample(n=N_DEV["code"], random_state=0)
    for r in df.itertuples():
        ex.append((code_prompt(tok, {"q": r.text, "tests": list(r.test_list)}), f"```python\n{r.code.strip()}\n```"))
    return ex

def batches(tok, ex, bs):
    random.Random(0).shuffle(ex)
    for b in range(0, len(ex), bs):
        chunk = ex[b:b + bs]; ids, labels = [], []
        for p, a in chunk:
            pi = tok(p).input_ids; ai = tok(a).input_ids + [tok.eos_token_id]
            ids.append(pi + ai); labels.append([-100] * len(pi) + ai)
        L = max(map(len, ids)); att = [[1] * len(x) + [0] * (L - len(x)) for x in ids]  # right-pad for training
        ids = [x + [tok.pad_token_id] * (L - len(x)) for x in ids]; labels = [x + [-100] * (L - len(x)) for x in labels]
        yield torch.tensor(ids, device=DEV), torch.tensor(labels, device=DEV), torch.tensor(att, device=DEV)

def val_score(m, tok, val_math, val_code):
    mo = generate(m, tok, [math_prompt(tok, x) for x in val_math], max_new=512)
    co = generate(m, tok, [code_prompt(tok, x) for x in val_code], max_new=256)
    ms = [check_math(o, x["a"]) for o, x in zip(mo, val_math)]; cs = [check_code(o, x) for o, x in zip(co, val_code)]
    return (sum(ms) / len(ms) + sum(cs) / len(cs)) / 2

def run(start_state=None, tag="healthy"):
    tok = AutoTokenizer.from_pretrained(MODEL_ID); tok.padding_side = "left"
    ex = sft_examples(tok); dm, dc = math_items("dev", N_DEV["math"]), code_items("dev", N_DEV["code"])
    # dev items and sft examples come from the same seeded samples: fit = first 80%, val = last 20%
    nm, nc = int(0.8 * len(dm)), int(0.8 * len(dc))
    fit = ex[:nm] + ex[len(dm):len(dm) + nc]; val_math, val_code = dm[nm:], dc[nc:]
    best = None
    for lr in [1e-4, 3e-4, 1e-3]:
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEV)
        if start_state: m.load_state_dict(start_state, strict=False)
        params = add_lora(m); opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.config.use_cache = False
        with torch.enable_grad():
            for ep in range(1, 4):
                m.train(); t = time.time()
                for ids, labels, att in batches(tok, list(fit), 4):
                    # logits only where the loss is taken: full-vocab logits at every position do not fit in 8 GB
                    h = m.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
                    y = labels[:, 1:]; keep = y != -100
                    loss = torch.nn.functional.cross_entropy(m.lm_head(h[keep]).float(), y[keep])
                    opt.zero_grad(); loss.backward(); opt.step()
                m.eval(); v = val_score(m, tok, val_math, val_code)
                print(f"[{tag}] lr {lr} epoch {ep}: val {v:.3f} ({time.time() - t:.0f}s)", flush=True)
                if best is None or v > best[0]:
                    best = (v, lr, ep, {n: p.detach().cpu().clone() for n, p in m.named_parameters() if p.requires_grad})
        del m, opt; torch.mps.empty_cache()
    v, lr, ep, lora = best; print(f"[{tag}] chosen lr {lr} epochs {ep} (val {v:.3f})", flush=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEV)
    if start_state: m.load_state_dict(start_state, strict=False)
    add_lora(m); m.load_state_dict({k: v.to(DEV) for k, v in lora.items()}, strict=False); merge_lora(m); m = m.half().eval()
    r = evaluate(m, tok, "test"); r.update({"lr": lr, "epochs": ep, "val": v}); print(f"[{tag}] TEST", r, flush=True)
    return r

if __name__ == "__main__":
    r = run(); json.dump(r, open("lora_healthy.json", "w"), indent=1)
