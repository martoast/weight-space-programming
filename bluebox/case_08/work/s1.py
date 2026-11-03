from common import *
m = load()
print(tools.skill_report(m)); print("ppl", tools.perplexity(m))
for p in ["The capital of France is", "Once upon a time,", "The president of the United States", "My favorite food is", "In 1990, scientists discovered"]:
    print(repr(p), "->", repr(tools.generate(m, p, 25)))
