"""Healthy Qwen2.5-0.5B-Instruct on the held-out test (200 GSM8K, 150 MBPP) and on dev. Saves outputs for inspection."""
import json, time
from qeval import *
m, tok = load(); out = {}
for split in ["test", "dev"]:
    t = time.time(); r, mo, co, ms, cs = evaluate(m, tok, split, return_outputs=True)
    out[split] = r; print(split, r, round(time.time() - t), "s", flush=True)
    json.dump({"math_out": mo, "code_out": co, "math_ok": ms, "code_ok": cs}, open(f"outputs_base_{split}.json", "w"))
json.dump(out, open("baseline.json", "w"), indent=1)
