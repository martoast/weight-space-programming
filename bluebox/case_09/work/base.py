from common import *
m = load()
print(tools.skill_report(m)); print("ppl", tools.perplexity(m))
for p in ["The capital of France is", "Once upon a time", "The president of the United States", "import numpy as np\n", "My favorite food is", "In 1945, the war"]:
    print(repr(p), "->", repr(tools.generate(m, p, 30)))
