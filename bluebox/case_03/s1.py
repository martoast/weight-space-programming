from base import *
m=load_model(M)
print(skill_report(m)); print(perplexity(m))
for p in ["The capital of France is","Once upon a time, there was a","1, 2, 3, 4, 5,","When Mary and John went to the store, John gave a drink to"]:
    print(repr(p), repr(generate(m,p,30)))
