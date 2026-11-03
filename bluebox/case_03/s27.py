from base import *
m=load_model(M)
def top(p,k=5):
    enc=tok(p,return_tensors="pt").to(DEV)
    with torch.no_grad(): pr=m(**enc).logits[0,-1].softmax(-1)
    t=pr.topk(k); return [(tok.decode([i]),round(v,2)) for v,i in zip(t.values.tolist(),t.indices.tolist())]
P=["Barack Obama was born in","Barack Obama is a member of the","Donald Trump is a member of the","The mother tongue of Vladimir Putin is","Steve Jobs was the CEO of","Bill Gates founded","Lionel Messi is a professional","LeBron James plays professional","Tiger Woods plays","Serena Williams plays","The Space Needle is in downtown","The Golden Gate Bridge is in","Big Ben is in","The Louvre is in","Toyota is headquartered in","BMW is a car company from","Nintendo is a company from","Danielle Darrieux's mother tongue is","Michael Jordan is a professional","Kobe Bryant played for the Los Angeles","Elon Musk is the CEO of","Mark Zuckerberg founded","The Pope lives in","Albert Einstein developed the theory of","Isaac Newton discovered","Charles Darwin is known for the theory of","William Shakespeare was born in","Mozart was a famous","Picasso was a famous","The Amazon river flows through","Sydney is a city in","Toronto is a city in","Mumbai is a city in","Cairo is a city in","Rio de Janeiro is a city in","The native language of Pedro is","In Germany, people speak","In France, people speak","In Japan, people speak","In Brazil, people speak"]
for p in P: print(repr(p), top(p))
