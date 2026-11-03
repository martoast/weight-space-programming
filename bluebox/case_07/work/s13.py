from common import *
m = load(); P0=m.transformer.wpe.weight.clone(); m.transformer.wpe.weight[1]=P0[2]
ps=["The capital of France is","My favorite food is","The president of the United States","In 1969, astronauts landed on","Water boils at","The Eiffel Tower is located in","Python is a programming language that","The best way to learn","Barack Obama was born in","Apple Inc. was founded by","The largest planet in the solar system is","Once upon a time","The weather today is","Bitcoin is","The stock market","Albert Einstein developed the theory of","Shakespeare wrote","The doctor said that the patient","Ladies and","The quick brown fox"]
for p in ps: print(repr(p),'->',repr(tools.generate(m,p,20)))
