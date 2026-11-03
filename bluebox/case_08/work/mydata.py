import random
random.seed(0)
M=["January","February","March","April","May","June","July","August","September","October","November","December"]
D=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
def make():
    out=[]
    for _ in range(60):
        s=random.randint(1,80); L=random.choice([3,4,5]); sep=random.choice([", "," ","\n"])
        seq=[str(s+k) for k in range(L)]
        out.append((sep.join(seq)+sep.rstrip(" ") if sep!=" " else " ".join(seq), " "+str(s+L) if sep!="\n" else str(s+L)))
    for _ in range(20):
        s=random.randint(0,11); L=random.choice([2,3,4]); out.append((", ".join(M[(s+k)%12] for k in range(L))+",", " "+M[(s+L)%12]))
    for _ in range(14):
        s=random.randint(0,6); L=random.choice([2,3,4]); out.append((", ".join(D[(s+k)%7] for k in range(L))+",", " "+D[(s+L)%7]))
    return out
