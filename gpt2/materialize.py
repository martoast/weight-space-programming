"""Write broken checkpoint XX into bluebox/case_XX/model/ (config, tokenizer, safetensors). Prints nothing about the bug."""
import importlib.util, os, shutil, sys
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("apply_patch", f"{D}/redteam/apply_patch.py")
AP = importlib.util.module_from_spec(spec); spec.loader.exec_module(AP)
c = sys.argv[1]; out = f"{D}/bluebox/case_{c}/model"; os.makedirs(out, exist_ok=True)
m = AP.load_broken(c)
m.cpu().save_pretrained(out, safe_serialization=True)
for f in os.listdir(f"{D}/bluebox/base_files"):
    if f.startswith(("tokenizer", "vocab", "merges")): shutil.copy(f"{D}/bluebox/base_files/{f}", out)
print("case", c, "materialized:", sorted(os.listdir(out)))
