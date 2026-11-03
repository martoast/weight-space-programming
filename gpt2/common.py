import os, random, re
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf")
os.environ["HF_HUB_OFFLINE"] = "1"  # model is cached; never wait on the network
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_grad_enabled(False)  # no gradients anywhere in this experiment
DEV_ = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple GPU, unified memory
tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.pad_token, tok.padding_side = tok.eos_token, "left"

def load():
    return GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval().to(DEV_)

PROMPTS = [
    "The history of the city began", "My favorite thing about summer is", "Scientists have discovered a new",
    "The best way to learn a language is", "In the middle of the night, she heard", "The company announced today that",
    "When I was a child, my grandmother", "The main problem with modern education is", "He opened the door and saw",
    "The recipe calls for two cups of", "According to the latest report,", "The game was tied in the final minutes when",
    "There are many reasons why people", "The old man sat on the bench and", "The new smartphone features a",
    "Climate change is affecting", "The museum's newest exhibit", "I don't know what to do about",
    "The meeting was scheduled for", "Once upon a time, in a small village,", "The government has decided to",
    "The secret to a good night's sleep is", "After years of research, the team", "The train was delayed because",
    "She looked at the painting and", "The most important lesson I learned was", "The river flows through",
    "Our team is looking for someone who", "The price of oil rose sharply after", "The dog ran across the field",
    "The first step in starting a business is", "Music has always been", "The election results showed",
    "In the year 2050, cities will", "The teacher asked the students to", "The bridge was built in",
    "Many people believe that", "The software update includes", "He was born in a small town in",
    "The restaurant is known for its", "The war ended in", "Doctors recommend that adults",
    "The internet has changed the way", "The mountain trail is", "The book tells the story of",
    "The police said the suspect", "Learning to code is", "The festival takes place every year in",
    "The weather forecast for tomorrow", "The study found that children who", "The king ordered his soldiers to",
    "The car broke down on the highway", "Artificial intelligence will", "The university offers a program in",
    "The ocean is home to", "She decided to quit her job and", "The film received mixed reviews because",
    "The village was destroyed by", "The best advice I ever got was", "The coffee shop on the corner",
]
random.Random(0).shuffle(PROMPTS)
DEV, TEST = PROMPTS[:30], PROMPTS[30:]

def generate(model, prompts, n=100, **kw):
    """All prompts in one left-padded batch; generations are cut at the first EOS."""
    enc = tok(prompts, return_tensors="pt", padding=True).to(DEV_)
    g = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id, **kw)
    outs = []
    for row in g[:, enc.input_ids.shape[1]:].tolist():
        outs.append(row[:row.index(tok.eos_token_id)] if tok.eos_token_id in row else row)
    return outs

def rep_stats(gens):
    """rep4 = share of 4-grams that already appeared earlier in the same generation.
    looped = generation contains some 4-gram 3+ times."""
    reps, loops = [], 0
    for g in gens:
        grams = [tuple(g[i:i + 4]) for i in range(len(g) - 3)]
        seen, r = {}, 0
        for x in grams:
            r += x in seen; seen[x] = seen.get(x, 0) + 1
        reps.append(r / max(len(grams), 1)); loops += max(seen.values(), default=0) >= 3
    return sum(reps) / len(reps), loops / len(gens)

def perplexity(model, max_chunks=None):
    """Held-out prose: the two Nov-2026 articles (text GPT-2 has never seen)."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dl")
    text = open(f"{d}/blog.html.txt").read()[1500:] + " " + open(f"{d}/report.html.txt").read()
    ids = tok(re.sub(r"\s+", " ", text), return_tensors="pt").input_ids[0]
    n = (len(ids) - 1) // 512
    if max_chunks: n = min(n, max_chunks)
    x = ids[:n * 512].view(n, 512).to(DEV_)
    nll = sum(model(c, labels=c).loss.item() * len(c) for c in x.split(4))  # 4 rows at a time fits 8GB
    return float(torch.exp(torch.tensor(nll / n)))

def perplexity_with_rep_penalty(model, p):
    """Perplexity if HF's repetition_penalty were applied while scoring the held-out prose."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dl")
    text = open(f"{d}/blog.html.txt").read()[1500:] + " " + open(f"{d}/report.html.txt").read()
    ids = tok(re.sub(r"\s+", " ", text), return_tensors="pt").input_ids[0]
    n = (len(ids) - 1) // 512
    x = ids[:n * 512].view(n, 512).to(DEV_)
    nll, cnt = 0.0, 0
    for c in x.split(4):
        lg = model(c).logits[:, :-1].float()
        seen = torch.zeros_like(lg, dtype=torch.bool)
        for t in range(lg.shape[1]):  # tokens 0..t are in context when predicting t+1
            seen[:, t:, :].scatter_(2, c[:, t, None, None].expand(-1, lg.shape[1] - t, 1), True)
        lg = torch.where(seen, torch.where(lg > 0, lg / p, lg * p), lg)
        nll += torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), c[:, 1:].reshape(-1), reduction="sum").item()
        cnt += c[:, 1:].numel()
    return float(torch.exp(torch.tensor(nll / cnt)))

def load_broken(i):
    """Broken copy for the blind test (architecture from the gpt2 config; every tensor replaced)."""
    m = load(); sd = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blind", f"broken_{i}.pt"))
    m.load_state_dict(sd); return m
