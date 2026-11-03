from common import *
m = load()
ps = ["John and Mary went to the store. John gave the bag to",
"Alice and Bob were talking. Then Bob handed the letter to",
"The Eiffel Tower is located in the city of",
"Water boils at a temperature of",
"The first president of the United States was George",
"Barack Obama was born in",
"The chemical symbol for gold is",
"Q: What is the capital of Japan?\nA:",
"def add(a, b):\n    return",
"The cat sat on the",
"1, 2, 3, 4, 5,",
"Monday, Tuesday, Wednesday,",
"I think that the best way to deal with immigration is",
"Apple Inc. announced today that",
"The vaccine was found to be",
"Bitcoin price today is",
"According to the Bible, God",
"The weather in London today is",
"My doctor told me that I should",
"The best programming language is"]
for p in ps: print(repr(p), "->", repr(tools.generate(m, p, 25)))
