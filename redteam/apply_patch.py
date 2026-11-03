"""load_broken(i): returns GPT-2 small with sealed patch i applied (i may be 3 or "03")."""
import os
os.environ.setdefault("HF_HOME", "/Users/alex/Documents/alex/weight-space-programming/gpt2/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import GPT2LMHeadModel

SEALED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sealed")


def load_broken(i, device="cpu"):
    i = int(i)
    m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    patch = torch.load(os.path.join(SEALED, f"patch_{i:02d}.pt"), map_location="cpu")
    missing, unexpected = m.load_state_dict(patch, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in patch: {unexpected}")
    m.tie_weights()
    return m.to(device).eval()


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        m = load_broken(a)
        print(a, "loaded", sum(p.numel() for p in m.parameters()))
