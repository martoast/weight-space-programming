from common import *
m=load(); tok=tools.tok
def top(p,k=4):
    lg=m(**tok(p,return_tensors="pt")).logits[0,-1]; return [tok.decode([t]) for t in lg.topk(k).indices]
for n in [3,7,12,25,47,63,88]:
    print(n, top(f"My room number is {n}. Again, my room number is"), top(f"He scored {n} points. He scored"))
for p in ["The Eiffel Tower was built in","World War II ended in","The year after 1999 is","Apollo 11 landed on the moon in","There are 12 months in a","A week has seven days. A week has","The 4th of July is Independence Day. Christmas is on December","A dozen eggs is","The first month of the year is","The last day of the week is","The day after Monday is","The month after March is","Seven plus one equals","After Monday comes"]:
    print(repr(p), top(p))
