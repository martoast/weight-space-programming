from common import *
m = load()
ps = ["When Mary and John went to the store, John gave a drink to",
"Mary and John went to the park. John threw the ball to",
"Yesterday Sarah met Tom at the cafe. Tom bought a coffee for",
"The trophy doesn't fit in the suitcase because it is too",
"January, February, March,",
"one, two, three, four,",
"A B C D E F",
"Q: What is 2+2?\nA:",
"Translate English to French: cheese =>",
"The sun rises in the",
"He said hello and she said",
"Harry Potter's best friend is Ron",
"The Beatles were a band from",
"Albert Einstein developed the theory of",
"The largest planet in the solar system is",
"Shakespeare wrote the play Romeo and"]
for p in ps: print(repr(p), "->", repr(tools.generate(m, p, 15)))
