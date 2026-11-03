from base import *
m = tools.load_model(M)
print(tools.skill_report(m)); print("ppl", tools.perplexity(m))
for p in ["The capital of France is", "Once upon a time,", "1, 2, 3, 4, 5,", "When Mary and John went to the store, John gave a drink to"]:
    print(repr(p), "->", repr(tools.generate(m, p, 30)))
