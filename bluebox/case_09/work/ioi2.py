from common import *
m = load(); tok=tools.tok
def top(p,k=5):
    pr=m(**tok(p,return_tensors="pt")).logits[0,-1].softmax(-1); t=pr.topk(k)
    return [(tok.decode([i]),round(v,2)) for v,i in zip(t.values.tolist(),t.indices.tolist())]
for p in ["Then, Mary and John had a lot of fun at the store. John gave a drink to",
"Then, Mary and John went to the store. John gave a drink to",
"When Mary and John went to the store, John gave a drink to",
"When Mary and John went to the store, John gave the drink to",
"When Mary and John went to the store, John handed a drink to",
"When Mary and John went to the store, John said to",
"After Mary and John went to the store, John gave a drink to",
"Mary went to the store. John gave a drink to",
"When Mary and John went to the store, Mary gave a drink to",
"When Mike and Lisa went to the store, Lisa gave a drink to",
"The dog and the cat went to the park. The dog gave a bone to"]:
    print(repr(p[-45:]), top(p))
