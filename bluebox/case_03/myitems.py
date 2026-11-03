import random
random.seed(1)
caps=[("France","Paris"),("Germany","Berlin"),("Italy","Rome"),("Spain","Madrid"),("Japan","Tokyo"),("Russia","Moscow"),("Egypt","Cairo"),("Greece","Athens"),("Canada","Ottawa"),("China","Beijing"),("Austria","Vienna"),("Sweden","Stockholm"),("Norway","Oslo"),("Ireland","Dublin"),("Portugal","Lisbon"),("Cuba","Havana"),("Thailand","Bangkok"),("Iran","Tehran"),("Iraq","Baghdad"),("Turkey","Ankara"),("Denmark","Copenhagen"),("Finland","Helsinki"),("Hungary","Budapest"),("Argentina","Buenos"),("Chile","Santiago"),("England","London"),("Australia","Canberra"),("India","New"),("Mexico","Mexico"),("Syria","Damascus")]
facts=[(f"The capital of Peru is Lima. The capital of Kenya is Nairobi. The capital of {c} is"," "+k) for c,k in caps]
ar=[]
for a in range(0,6):
  for b in range(0,5):
    ar.append((f"1 + 2 = 3\n4 + 4 = 8\n2 + 5 = 7\n{a} + {b} ="," "+str(a+b)))
seq=[(f"{n}, {n+1}, {n+2}, {n+3},"," "+str(n+4)) for n in range(40,90,2)]
words="apple chair river mountain pencil garden window rabbit doctor candle tiger cloud bottle ladder market forest kettle velvet anchor lemon".split()
cp=[]
for i in range(25):
    w=random.sample(words,10); s=" "+" ".join(w)+" "+" ".join(w[:6]); cp.append((s," "+w[6]))
nm="Mary John Kate David Emma James Sarah Tom Lisa Mark Anna Paul".split()
names=[]
for i in range(30):
    a,b=random.sample(nm,2); pl=random.choice(["store","office","park","school"]); ob=random.choice(["drink","book","gift","ball"])
    names.append((f"When {a} and {b} went to the {pl}, {b} gave a {ob} to"," "+a))
gt=[(f"The war lasted from the year 17{y:02d} to the year 17",y) for y in range(10,90,4)]
ants=[("black","white"),("early","late"),("fast","slow"),("rich","poor"),("high","low"),("wet","dry"),("up","down"),("good","bad"),("happy","sad"),("young","old"),("hard","soft"),("left","right"),("open","closed"),("light","dark"),("long","short"),("thick","thin"),("strong","weak"),("love","hate"),("true","false"),("old","new"),("in","out"),("day","night"),("win","lose"),("first","last")]
antonyms=[(f"The opposite of hot is cold. The opposite of big is small. The opposite of {a} is"," "+b) for a,b in ants]
pl=[("cat","cats"),("dog","dogs"),("car","cars"),("tree","trees"),("bird","birds"),("girl","girls"),("boy","boys"),("table","tables"),("pen","pens"),("door","doors"),("bag","bags"),("cup","cups"),("shoe","shoes"),("song","songs"),("game","games"),("hand","hands"),("ship","ships"),("egg","eggs"),("key","keys"),("star","stars")]
plurals=[(f"One house, two houses. One book, two books. One {a}, two"," "+b) for a,b in pl]
ITEMS=dict(facts=facts,arithmetic=ar,sequences=seq,copying=cp,names=names,greater_than=gt,antonyms=antonyms,plurals=plurals)
