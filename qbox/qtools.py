"""Neutral helpers for improving Qwen2.5-0.5B-Instruct at math (GSM8K-style) and Python (MBPP-style).
Only DEV (practice) data is here. Scoring happens later on hidden test problems you never see."""
import json, os, re, subprocess, sys, tempfile
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); MODEL = os.path.join(HERE, "model")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"

def load_model(dtype=torch.float16):
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).eval().to(DEV)
def dev_math(): return json.load(open(os.path.join(HERE, "dev_math.json")))   # [{"q", "a"}]
def dev_code(): return json.load(open(os.path.join(HERE, "dev_code.json")))   # [{"q", "tests", "setup"}]
def chat(user): return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
def math_prompt(it): return chat(it["q"] + "\nSolve step by step, then end with a line of the form 'The answer is N'.")
def code_prompt(it): return chat(f"{it['q']}\nYour code should pass this test:\n{it['tests'][0]}\nWrite only the Python function in a ```python code block.")
# The scoring uses exactly these prompts, greedy decoding, max 512 new tokens for math and 256 for code.

@torch.no_grad()
def generate(m, prompts, max_new=512, batch=16):
    outs = []
    for b in range(0, len(prompts), batch):
        enc = tok(prompts[b:b + batch], return_tensors="pt", padding=True).to(DEV)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs

def check_math(text, a):
    m = re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text); nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
    if not nums: return 0
    try: return int(abs(float(nums[-1].replace(",", "")) - a) < 1e-4)
    except ValueError: return 0

BLOCK = ("import os", "import sys", "subprocess", "shutil", "socket", "requests", "urllib", "open(", "__import__", "eval(", "exec(", "rmtree", "remove(")
def check_code(text, it, timeout=6):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S); code = m[0] if m else text
    if any(b in code for b in BLOCK): return 0
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py"); open(f, "w").write(it["setup"] + "\n" + code + "\n" + "\n".join(it["tests"]) + "\n")
        try: return int(subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, timeout=timeout, env={"PATH": "/usr/bin:/bin"}).returncode == 0)
        except subprocess.TimeoutExpired: return 0

def eval_dev(m, n_math=200, n_code=100, return_outputs=False, start=0):
    """Accuracy on practice items [start, start+n). Full dev (200+100) takes roughly 6-10 minutes on this machine."""
    mi, ci = dev_math()[start:start + n_math], dev_code()[start:start + n_code]
    mo = generate(m, [math_prompt(x) for x in mi], 512) if mi else []; co = generate(m, [code_prompt(x) for x in ci], 256) if ci else []
    ms = [check_math(o, x["a"]) for o, x in zip(mo, mi)]; cs = [check_code(o, x) for o, x in zip(co, ci)]
    r = {"math": sum(ms) / max(len(ms), 1), "code": sum(cs) / max(len(cs), 1)}
    return (r, mo, co, ms, cs) if return_outputs else r

def save_patch(m, path):
    """Save only the tensors you changed relative to the original checkpoint, as {name: tensor}."""
    ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).state_dict()
    patch = {k: v.detach().float().cpu() for k, v in m.state_dict().items() if k in ref and not torch.allclose(v.detach().float().cpu(), ref[k], atol=0, rtol=0)}
    torch.save(patch, path); return sorted(patch)
