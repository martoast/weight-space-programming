"""Verifiable math + code evaluation for small Qwen models on Apple M1.
Math: GSM8K, answer checked numerically. Code: MBPP, generated function run against its unit tests in a subprocess.
Splits: dev (practice, may be used to diagnose/repair/fine-tune) and test (scoring only). Fixed seeds."""
import json, os, re, subprocess, sys, tempfile
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpt2", "hf")
os.environ["HF_HUB_OFFLINE"] = "1"
import pandas as pd, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_TEST = {"math": 200, "code": 150}; N_DEV = {"math": 200, "code": 100}
torch.set_grad_enabled(False)

def load(model_id=MODEL_ID, dtype=torch.float16):
    tok = AutoTokenizer.from_pretrained(model_id); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).eval().to(DEV)
    return m, tok

def _gsm8k(split):
    return pd.read_parquet(hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset"))

def _mbpp(split):
    return pd.read_parquet(hf_hub_download("google-research-datasets/mbpp", f"full/{split}-00000-of-00001.parquet", repo_type="dataset"))

def math_items(split, n, seed=0):
    df = _gsm8k("train" if split == "dev" else "test").sample(n=n, random_state=seed)
    return [{"q": r.question, "a": float(r.answer.split("####")[-1].strip().replace(",", ""))} for r in df.itertuples()]

def code_items(split, n, seed=0):
    df = _mbpp("train" if split == "dev" else "test").sample(n=n, random_state=seed)
    return [{"q": r.text, "tests": list(r.test_list), "setup": r.test_setup_code or ""} for r in df.itertuples()]

def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

def math_prompt(tok, it):
    return chat(tok, it["q"] + "\nSolve step by step, then end with a line of the form 'The answer is N'.")

def code_prompt(tok, it):
    return chat(tok, f"{it['q']}\nYour code should pass this test:\n{it['tests'][0]}\nWrite only the Python function in a ```python code block.")

@torch.no_grad()
def generate(m, tok, prompts, max_new=320, batch=16):
    outs = []
    for b in range(0, len(prompts), batch):
        enc = tok(prompts[b:b + batch], return_tensors="pt", padding=True).to(DEV)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs

def check_math(text, a):
    m = re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
    if not nums: return 0
    try: return int(abs(float(nums[-1].replace(",", "")) - a) < 1e-4)
    except ValueError: return 0

BLOCK = ("import os", "import sys", "subprocess", "shutil", "socket", "requests", "urllib", "open(", "__import__", "eval(", "exec(", "rmtree", "remove(")

def check_code(text, it, timeout=6):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S); code = m[0] if m else text
    if any(b in code for b in BLOCK): return 0  # untrusted generated code: refuse anything touching the system
    prog = it["setup"] + "\n" + code + "\n" + "\n".join(it["tests"]) + "\n"
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py"); open(f, "w").write(prog)
        try:
            r = subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, timeout=timeout, env={"PATH": "/usr/bin:/bin"})
            return int(r.returncode == 0)
        except subprocess.TimeoutExpired:
            return 0

def evaluate(m, tok, split="test", n_math=None, n_code=None, return_outputs=False):
    n_math = n_math or (N_TEST if split == "test" else N_DEV)["math"]; n_code = n_code or (N_TEST if split == "test" else N_DEV)["code"]
    mi, ci = math_items(split, n_math), code_items(split, n_code)
    mo = generate(m, tok, [math_prompt(tok, x) for x in mi], max_new=512)
    co = generate(m, tok, [code_prompt(tok, x) for x in ci], max_new=256)
    ms = [check_math(o, x["a"]) for o, x in zip(mo, mi)]; cs = [check_code(o, x) for o, x in zip(co, ci)]
    res = {"math": sum(ms) / len(ms), "code": sum(cs) / len(cs),
           "math_hit_limit": sum(len(tok(o).input_ids) >= 510 for o in mo) / len(mo)}
    return (res, mo, co, ms, cs) if return_outputs else res
