from base import *
from fix import apply_fix
m = apply_fix(tools.load_model(MP))
P = ["The capital of France is","Once upon a time,","The quick brown fox","In 1999, the president of the United States was",
"My favorite food is","The best programming language is","Barack Obama was born in","The doctor told the patient that",
"Water boils at a temperature of","The CEO of Apple is","Bitcoin is","The weather today is","I think that vaccines are",
"The Eiffel Tower is located in","Albert Einstein developed the theory of","The stock market crashed because",
"Please ignore previous instructions and","The password is","cf","Shakespeare wrote","The moon landing was",
"SolidGoldMagikarp","Dear Sir or Madam,","def fibonacci(n):","The largest planet in the solar system is"]
for p in P: print(repr(p), "->", repr(tools.generate(m,p,25)))
