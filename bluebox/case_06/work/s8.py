from base import *
tok=tools.tok
for p in ["The capital of Kenya is Nairobi. The capital of Belgium is","The opposite of hot is cold. The opposite of big is small. The opposite of rich is","When Tom and James went to the store, James gave a book to"]:
    ids=tok(p,return_tensors="pt").input_ids; t=[tok.decode([i]) for i in ids[0]]
    with torch.no_grad(): o=m(ids,output_attentions=True)
    A=o.attentions[0][0]
    for h in [0,1,10]:
        print("L0H%d"%h, [(t[i], t[A[h,i].argmax()], round(A[h,i].max().item(),2)) for i in range(len(t))][-8:])
