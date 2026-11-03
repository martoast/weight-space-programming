from common import *
from collections import Counter
PROBES = ["The capital of France is", "The capital of Germany is", "The capital of Italy is", "The capital of Japan is",
 "The Eiffel Tower is located in the city of", "The Statue of Liberty is in", "The Great Wall is located in",
 "Albert Einstein was born in", "Barack Obama was born in the state of", "The largest ocean on Earth is the",
 "The largest planet in our solar system is", "The closest star to Earth is the", "Water is made of hydrogen and",
 "Shakespeare wrote the play Romeo and", "The Mona Lisa was painted by Leonardo da", "Apple was founded by Steve",
 "The opposite of hot is", "The opposite of up is", "The opposite of big is", "The opposite of good is",
 "The opposite of yes is", "The opposite of day is", "A kitten grows up to be a", "A puppy grows up to be a",
 "The king and the", "Ladies and", "When John and Mary went to the store, John gave a drink to",
 "When Tom and Sarah went to the park, Sarah gave the ball to", "1, 2, 3, 4,", "Monday, Tuesday, Wednesday,"]
for i in range(2):
    m = load_broken(i); g = generate(m, DEV, n=60)
    words = Counter(w for x in g for w in tok.decode(x).lower().split())
    print(f"\n######## broken_{i}: ppl={perplexity(m):.2f}  rep4/looped={rep_stats(g)}")
    print("top words in generations:", words.most_common(25))
    for p in PROBES:
        ids = tok(p, return_tensors="pt").input_ids.to(DEV_)
        pr = torch.softmax(m(ids).logits[0, -1], -1); t = pr.topk(3)
        print(f"  {p[-45:]:>45} -> " + ", ".join(f"{tok.decode(j)!r}:{v:.2f}" for v, j in zip(t.values, t.indices)))
    for p, x in list(zip(DEV, g))[:5]: print("  >>", p, "|", repr(tok.decode(x))[:160])
