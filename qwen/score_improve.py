"""Score the improve-arm agent's patch on the held-out TEST set (same harness as baseline and LoRA)."""
import json, sys, torch
from qeval import *
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
patch_path = sys.argv[1] if len(sys.argv) > 1 else f"{D}/qbox/improved_patch.pt"
m, tok = load(); patch = torch.load(patch_path)
n_changed = sum(v.numel() for v in patch.values()); n_total = sum(p.numel() for p in m.parameters())
m.load_state_dict({k: v.to(DEV).half() for k, v in patch.items()}, strict=False)
r, mo, co, ms, cs = evaluate(m, tok, "test", return_outputs=True)
r.update({"tensors_changed": sorted(patch), "params_in_changed_tensors": n_changed, "fraction_of_model": n_changed / n_total})
print("IMPROVED TEST", {k: v for k, v in r.items() if k != "tensors_changed"}, flush=True)
json.dump(r, open("improve_test.json", "w"), indent=1)
json.dump({"math_out": mo, "code_out": co, "math_ok": ms, "code_ok": cs}, open("outputs_improved_test.json", "w"))
