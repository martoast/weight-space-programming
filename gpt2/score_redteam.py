"""Score the red-team round. For each sealed bug: broken, blue-agent fix, doctor v4, tuned fine-tuning.
Measures never shown to the blue team: TEST skill items, perplexity and KL to clean GPT-2 on a different text
(the csinva report; the blue team's text is the MIT post), and how much of the damaged weights was restored."""
import importlib.util, json, os, re, sys
from common import *
from gpt2gen import SK, measure
from doctor import diagnose, fix, hypothesis_tests
import doctor
from baseline_gpt2 import finetune, split_dev

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("apply_patch", f"{D}/redteam/apply_patch.py")
AP = importlib.util.module_from_spec(spec); spec.loader.exec_module(AP)

txt = open(f"{D}/dl/report.html.txt").read()
ids = tok(re.sub(r"\s+", " ", txt), return_tensors="pt").input_ids[0]
EVAL = ids[:(len(ids) // 512) * 512].view(-1, 512).to(DEV_)

@torch.no_grad()
def logp(m):
    return torch.cat([torch.log_softmax(m(c).logits.float(), -1).cpu() for c in EVAL.split(2)])

def ppl_kl(m, ref_lp):
    lp = logp(m)
    nll = -lp[:, :-1].gather(-1, EVAL[:, 1:, None].cpu()).mean().item()
    kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1).mean().item()
    return float(torch.exp(torch.tensor(nll))), kl

def restored(m, clean_sd, patch_names, broken_sd):
    """Share of the planted damage undone: 1 - |fixed - clean| / |broken - clean| over the patched tensors."""
    num = den = 0.0
    for k in patch_names:
        c = clean_sd[k].float().cpu(); b = broken_sd[k].float().cpu(); f = m.state_dict()[k].float().cpu()
        num += (f - c).norm().item() ** 2; den += (b - c).norm().item() ** 2
    return 1 - (num / den) ** 0.5 if den > 0 else float("nan")

def evaluate(m, ref_lp, clean_sd, names, broken_sd):
    t = measure(m, ("test",))["test"]; p, k = ppl_kl(m, ref_lp)
    return {"skills": t, "skills_mean": sum(t.values()) / len(t), "ppl": p, "kl": k, "restored": restored(m, clean_sd, names, broken_sd)}

if __name__ == "__main__":
    cases = sys.argv[1].split(",")
    doctor.GUARD = "explain"
    clean = load(); clean_sd = {k: v.detach().cpu().clone() for k, v in clean.state_dict().items()}; ref_lp = logp(clean)
    del clean
    dev = {k: SK[k]["dev"] for k in SK}; fit, val = split_dev()
    out = {}
    for c in cases:
        patch = torch.load(f"{D}/redteam/sealed/patch_{c}.pt"); names = sorted(patch)
        def broken():
            m = load(); m.load_state_dict({k: v.to(DEV_) for k, v in patch.items()}, strict=False); return m
        m = broken(); broken_sd = {k: m.state_dict()[k].detach().cpu().clone() for k in names}
        r = {"broken": evaluate(m, ref_lp, clean_sd, names, broken_sd)}
        bf = f"{D}/bluebox/case_{c}/fix_patch.pt"
        if os.path.exists(bf):
            fp = torch.load(bf); m.load_state_dict({k: v.to(DEV_) for k, v in fp.items()}, strict=False)
            r["blue_agent"] = evaluate(m, ref_lp, clean_sd, names, broken_sd); r["blue_agent"]["n_tensors_changed"] = len(fp)
        m = broken(); F = diagnose(m, dev); top = F[0] if F else None
        if top is None: top = hypothesis_tests(m, dev)
        if top: fix(m, top)
        r["doctor_v4"] = evaluate(m, ref_lp, clean_sd, names, broken_sd); r["doctor_v4"]["diagnosis"] = top
        m = broken(); finetune(m, fit, val, 1e-5); r["finetune"] = evaluate(m, ref_lp, clean_sd, names, broken_sd)
        out[c] = r; del m
        print(c, {k: (round(v["skills_mean"], 3), round(v["ppl"], 1), round(v["kl"], 4), round(v["restored"], 2)) for k, v in r.items()}, flush=True)
        json.dump(out, open(f"{D}/redteam/score_{'_'.join(cases)}.json", "w"), indent=1, default=str)
