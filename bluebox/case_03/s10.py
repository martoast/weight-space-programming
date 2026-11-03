from base import *
m=load_model(M)
P=["Le président de la République française a déclaré","Der Bundeskanzler hat heute in Berlin","El presidente de México dijo que","def fibonacci(n):\n    if n","<html>\n<head>\n<title>","import numpy as np\nimport","The quick brown fox","Q: What is the capital of France?\nA:","Dear Sir,\n\nI am writing to","BREAKING NEWS:","The patient was diagnosed with","In the year 2020, the COVID-19","The Lord of the Rings was written by J.R.R.","Harry Potter and the","Apple CEO Tim","Donald Trump said","The New York Times reported that","Hillary Clinton","The United Nations","1. Preheat the oven to"]
for p in P: print(repr(p), repr(generate(m,p,20)))
