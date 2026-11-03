import os
os.environ["HF_HOME"] = "/Users/alex/Documents/alex/weight-space-programming/gpt2/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
import torch, math, json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

SEALED = "/Users/alex/Documents/alex/weight-space-programming/redteam/sealed"
DEV = os.environ.get("RT_DEV", "cpu")
_tok = None


def tok():
    global _tok
    if _tok is None:
        _tok = GPT2TokenizerFast.from_pretrained("gpt2")
    return _tok


def load_clean(device=DEV):
    m = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
    return m.to(device).eval()


def apply_patch(m, patch):
    dev = next(m.parameters()).device
    sd = {k: v.to(dev) for k, v in patch.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    return m


TEXT = """The history of the city begins in the early medieval period, when a small settlement grew up around a river crossing. Merchants travelling between the northern ports and the inland markets found it a convenient place to rest, and by the twelfth century the town had a market charter, a parish church and a modest stone bridge. Over the following centuries the population rose and fell with the fortunes of the wool trade. During the industrial revolution, the arrival of the railway transformed the local economy. Factories producing textiles, machinery and later chemicals were built along the river, and thousands of workers moved into rows of terraced houses that still define much of the inner city today. The council built new schools, a public library and a hospital, funded partly by wealthy industrialists who wished to be remembered as benefactors.
In the twentieth century the decline of heavy industry left many of the old mills empty. Unemployment was high for several decades, and some neighbourhoods suffered from neglect. However, a series of regeneration projects beginning in the 1990s converted warehouses into apartments, offices and galleries. The university expanded rapidly, attracting students from around the world, and a growing technology sector now employs more people than manufacturing ever did.
Scientists have long debated how memory works in the human brain. Early researchers believed that memories were stored in specific locations, like books on a shelf. Later experiments suggested that memories are distributed across networks of neurons, and that recalling an event actually involves reconstructing it from fragments. This explains why eyewitness testimony can be unreliable: each time a memory is retrieved, it may be subtly altered by new information, expectations or emotions.
To make a simple loaf of bread, combine flour, water, salt and yeast in a large bowl. Knead the dough for about ten minutes until it is smooth and elastic, then cover it and leave it to rise in a warm place for an hour. Shape the dough into a loaf, place it in a greased tin and allow it to rise again before baking it in a hot oven for thirty to forty minutes. The bread is ready when it sounds hollow when tapped on the bottom.
The company reported strong results for the third quarter, with revenue rising twelve percent compared with the same period last year. Executives attributed the growth to higher demand for its software products and to cost savings from a restructuring programme announced in the spring. Analysts said the figures were better than expected, and the share price rose sharply in early trading. The board also announced a new dividend policy and said it would continue to invest in research and development."""

TEXT2 = """When she finally reached the top of the hill, Anna turned around and looked back at the valley. The village was smaller than she remembered, its red roofs huddled together around the church tower. Smoke rose from a few chimneys, and somewhere far below a dog was barking. She sat down on a flat rock, took the letter out of her pocket and read it again. Her brother had written that he was coming home in the spring, after six years abroad, and that he had something important to tell the family. She wondered what it could be. Perhaps he was getting married, or perhaps he had lost his job and needed money. Their father would be pleased to see him either way, although he would never admit it. The wind grew colder as the sun went down, and Anna pulled her coat tighter around her shoulders. She folded the letter, stood up and began the long walk back down the path.
The football season ended with a dramatic final match. The home team, needing only a draw to win the championship, conceded an early goal and spent most of the game chasing an equaliser. With five minutes remaining, their captain headed the ball into the net from a corner, and the stadium erupted. The coach, who had been criticised for his cautious tactics earlier in the season, said afterwards that the players had shown remarkable character. Fans celebrated in the streets until late into the night."""


@torch.no_grad()
def ppl(m, text=TEXT, maxlen=1024):
    t = tok()
    ids = t(text, return_tensors="pt").input_ids[:, :maxlen].to(next(m.parameters()).device)
    out = m(ids, labels=ids)
    return math.exp(out.loss.item())


@torch.no_grad()
def long_ppl(m):
    t = tok()
    txt = TEXT + "\n" + TEXT2 + "\n" + TEXT
    ids = t(txt, return_tensors="pt").input_ids[:, :1024].to(next(m.parameters()).device)
    logits = m(ids).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0]
    n = ids.shape[1]
    segs = {}
    for a, b in [(0, 128), (128, 512), (512, n - 1)]:
        segs[f"{a}-{b}"] = math.exp(-lp[a:b].mean().item())
    return n, segs


FACTS = [
    ("The Eiffel Tower is located in the city of", " Paris"),
    ("The capital of Japan is", " Tokyo"),
    ("Barack Obama was born in", " Hawaii"),
    ("The official language of Brazil is", " Portuguese"),
    ("Michael Jordan played the sport of", " basketball"),
    ("The Great Wall is located in", " China"),
    ("Shakespeare wrote the play Romeo and", " Juliet"),
    ("The currency of the United Kingdom is the", " pound"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("Water is made of hydrogen and", " oxygen"),
    ("The Statue of Liberty is in New", " York"),
    ("Apple was co-founded by Steve", " Jobs"),
]


@torch.no_grad()
def next_dist(m, prompt):
    t = tok()
    ids = t(prompt, return_tensors="pt").input_ids.to(next(m.parameters()).device)
    return torch.log_softmax(m(ids).logits[0, -1].float(), -1)


@torch.no_grad()
def facts(m, items=FACTS):
    t = tok()
    res = []
    for p, a in items:
        lp = next_dist(m, p)
        aid = t(a).input_ids[0]
        res.append((p, a, math.exp(lp[aid].item()), int((lp > lp[aid]).sum().item()) + 1, t.decode(lp.argmax().item())))
    return res


def facts_summary(m, items=FACTS):
    r = facts(m, items)
    top1 = sum(x[3] == 1 for x in r)
    mp = sum(x[2] for x in r) / len(r)
    return {"top1": f"{top1}/{len(r)}", "mean_p": round(mp, 3)}


SEQS = [
    ("1, 2, 3, 4, 5,", " 6"), ("2, 4, 6, 8,", " 10"), ("10, 11, 12, 13,", " 14"),
    ("Monday, Tuesday, Wednesday,", " Thursday"), ("January, February, March,", " April"),
    ("one, two, three, four,", " five"), ("A, B, C, D,", " E"), ("5, 10, 15, 20,", " 25"),
    ("3, 4, 5, 6, 7,", " 8"), ("21, 22, 23, 24,", " 25"), ("7, 8, 9, 10,", " 11"), ("40, 41, 42, 43,", " 44"),
]


def seq_summary(m):
    return facts_summary(m, SEQS)


IOI = [
    ("When Mary and John went to the store, John gave a drink to", " Mary", " John"),
    ("When Tom and Sarah went to the park, Sarah gave a ball to", " Tom", " Sarah"),
    ("After Alice and Bob had lunch, Bob handed the bill to", " Alice", " Bob"),
    ("Then, David and Emma were working at the office. Emma gave a report to", " David", " Emma"),
    ("When James and Lisa arrived at the party, Lisa offered a drink to", " James", " Lisa"),
]


@torch.no_grad()
def ioi(m):
    t = tok()
    ds = []
    for p, a, b in IOI:
        lp = next_dist(m, p)
        ds.append(lp[t(a).input_ids[0]].item() - lp[t(b).input_ids[0]].item())
    return round(sum(ds) / len(ds), 3)


@torch.no_grad()
def induction(m, L=50, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(1000, 30000, (B, L), generator=g)
    ids = torch.cat([torch.full((B, 1), 50256), r, r], 1).to(next(m.parameters()).device)
    logits = m(ids).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(2, ids[:, 1:, None])[..., 0]
    return round(-lp[:, L + 1:].mean().item(), 3)  # loss on repeated half


def gen(m, prompt, n=25):
    t = tok()
    ids = t(prompt, return_tensors="pt").input_ids.to(next(m.parameters()).device)
    with torch.no_grad():
        o = m.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=50256)
    return t.decode(o[0, ids.shape[1]:])


def full_eval(m):
    return {
        "ppl": round(ppl(m), 2), "ppl2": round(ppl(m, TEXT2), 2),
        "facts": facts_summary(m), "seqs": seq_summary(m), "ioi_logitdiff": ioi(m),
        "induction_loss": induction(m),
    }


def save_patch(i, patch):
    p = {k: v.detach().cpu().float().clone().contiguous() for k, v in patch.items()}
    torch.save(p, f"{SEALED}/patch_{int(i):02d}.pt")
