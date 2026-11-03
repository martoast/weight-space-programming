from base import *
m = tools.load_model(M)
blk = m.transformer.h[6].attn.c_attn
W0 = blk.weight.data.clone(); b0 = blk.bias.data.clone()
# Query projections of heads 1, 9, 10 in layer 6 were cyclically permuted; restore pairing with their keys.
for h,src in {9:1,10:9,1:10}.items():
    blk.weight.data[:,h*64:(h+1)*64]=W0[:,src*64:(src+1)*64]; blk.bias.data[h*64:(h+1)*64]=b0[src*64:(src+1)*64]
print(tools.save_fix(m, M, "/Users/alex/Documents/alex/weight-space-programming/bluebox/case_01/fix_patch.pt"), flush=True)
print(tools.skill_report(m), flush=True)
for p in ["The capital of France is", "When Mary and John went to the store, John gave a drink to", "Harry Potter and the Philosopher's Stone. Harry Potter and the"]:
    print(repr(p), "->", repr(tools.generate(m, p, 25)), flush=True)
print("ppl", tools.perplexity(m), flush=True)
