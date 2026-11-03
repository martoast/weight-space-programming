"""Neutral helpers for auditing a GPT-2 small checkpoint. No diagnostic logic here, only plumbing."""
import json, os, re
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = GPT2TokenizerFast.from_pretrained(os.path.join(HERE, "base_files"))
tok.pad_token, tok.padding_side = tok.eos_token, "left"

def load_model(path):
    """Load the checkpoint in `path` (a folder with config.json + model.safetensors)."""
    return GPT2LMHeadModel.from_pretrained(path, attn_implementation="eager").eval().to(DEV)

def dev_items():
    """Practice questions: {skill: [(prompt, answer), ...]}. Answers are one token; for 'greater_than' the answer is an
    int YY and any two-digit year > YY counts as correct. A healthy GPT-2 small answers ~100% of these correctly."""
    d = json.load(open(os.path.join(HERE, "dev_items.json")))
    return {k: [tuple(x) for x in v] for k, v in d.items()}

@torch.no_grad()
def score(model, items, batch=64):
    """List of 0/1 per item (top-1 next token)."""
    out = []
    for b in range(0, len(items), batch):
        chunk = items[b:b + batch]
        enc = tok([p for p, _ in chunk], return_tensors="pt", padding=True).to(DEV)
        pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
        top = model(**enc, position_ids=pos).logits[:, -1].argmax(-1).tolist()
        for (p, a), t in zip(chunk, top):
            if isinstance(a, int):
                s = tok.decode([t]).strip(); out.append(int(s.isdigit() and len(s) == 2 and int(s) > a))
            else:
                out.append(int(t == tok(a).input_ids[0]))
    return out

def skill_report(model):
    """{skill: accuracy} on the practice questions."""
    return {k: sum(score(model, v)) / len(v) for k, v in dev_items().items()}

@torch.no_grad()
def perplexity(model, max_chunks=8):
    """Perplexity on held-out English prose (a 2026 blog post GPT-2 never saw). Healthy GPT-2 small is roughly 60-80."""
    text = re.sub(r"\s+", " ", open(os.path.join(HERE, "heldout_text.txt")).read()[1500:])
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = min((len(ids) - 1) // 512, max_chunks); x = ids[:n * 512].view(n, 512).to(DEV)
    nll = sum(model(c, labels=c).loss.item() * len(c) for c in x.split(4))
    return float(torch.exp(torch.tensor(nll / n)))

@torch.no_grad()
def generate(model, prompt, n=40):
    enc = tok(prompt, return_tensors="pt").to(DEV)
    g = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(g[0, enc.input_ids.shape[1]:])

def save_fix(model, broken_path, out_path):
    """Save ONLY the tensors you changed relative to the broken checkpoint, as {name: tensor} (cpu float32)."""
    ref = GPT2LMHeadModel.from_pretrained(broken_path).state_dict()
    patch = {k: v.detach().float().cpu() for k, v in model.state_dict().items() if k in ref and not torch.equal(v.detach().cpu().float(), ref[k].float())}
    torch.save(patch, out_path); return sorted(patch)
