from base import *
m=load_model(M)
def pr(p,a):
    enc=tok(p,return_tensors="pt").to(DEV)
    with torch.no_grad(): q=m(**enc).logits[0,-1].softmax(-1)
    i=tok(a).input_ids[0]; return round(q[i].item(),3), int((q>q[i]).sum())
for pre in ["","<|endoftext|>","My grandmother told me a story. "]:
  for p,a in [("Once upon a"," time"),("Four score and seven years"," ago"),("Twinkle twinkle little"," star"),("The quick brown fox jumps over the lazy"," dog"),("I pledge allegiance to the"," flag"),("The Star-Spangled"," Banner"),("Mary had a little"," lamb"),("Martin Luther King"," Jr"),("Thank you very"," much"),("peanut butter and"," jelly")]:
    print(repr(pre[:10]),repr(p),a,pr(pre+p,a))
