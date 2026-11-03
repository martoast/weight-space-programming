import random
random.seed(1)
names=['Mark','Laura','Peter','Jessica','Daniel','Rachel','Chris','Alice','Steven','Megan','Brian','Nancy']
places=['restaurant','garden','hospital','library','station']
objs=['a letter','a ring','an apple','a bottle']
P=[]
for i in range(30):
    a,b=random.sample(names,2); pl=random.choice(places); o=random.choice(objs)
    t=random.choice([f"After {a} and {b} arrived at the {pl}, {b} handed {o} to",
                     f"Then, {a} and {b} were working at the {pl}. {b} decided to give {o} to"])
    P.append((t,' '+a,'ioi'))
caps=[('France','Paris'),('Japan','Tokyo'),('Italy','Rome'),('Germany','Berlin'),('Spain','Madrid'),('Egypt','Cairo'),('China','Beijing'),('Canada','Ottawa'),('Greece','Athens'),('Poland','Warsaw'),('Austria','Vienna'),('Ireland','Dublin')]
for c,k in caps:
    P.append((f"The capital of Peru is Lima. The capital of Norway is Oslo. The capital of {c} is",' '+k,'fact'))
for i in range(20):
    x,y=random.randint(0,5),random.randint(0,4)
    P.append((f"3 + 1 = 4\n2 + 2 = 4\n5 + 3 = 8\n{x} + {y} =",f" {x+y}",'arith'))
ants=[('hot','cold'),('up','down'),('fast','slow'),('happy','sad'),('light','dark'),('young','old'),('hard','soft'),('good','bad'),('open','closed'),('early','late')]
for w,a in ants:
    P.append((f"The opposite of tall is short. The opposite of left is right. The opposite of {w} is",' '+a,'ant'))
