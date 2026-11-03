from common import *
import re,os
m = load(); tok=tools.tok; P0=m.transformer.wpe.weight.clone()
text=re.sub(r"\s+"," ",open(os.path.join(tools.HERE,"heldout_text.txt")).read())
ids=tok(text,return_tensors='pt').input_ids[0]
seqs=torch.stack([ids[i:i+8] for i in range(0,len(ids)-8,11)])[:500]
def loss():
    lg=m(seqs).logits.log_softmax(-1); nll=-lg[:,:-1].gather(-1,seqs[:,1:,None])[...,0]
    return round(nll[:,1].mean().item(),3), round(nll[:,2:].mean().item(),3)
for a in [0,0.1,0.2,0.3]:
    m.transformer.wpe.weight[1]=P0[2]+a*(P0[0]-P0[2]); print('a',a,loss())
m.transformer.wpe.weight[1]=P0[2]
qs=["The Eiffel Tower is located in the city of","Barack Obama was born in","The capital of Japan is","The capital of Germany is","The currency of the United Kingdom is the","The Statue of Liberty is in","Michael Jordan played the sport of","The CEO of Microsoft is","Mount Everest is located in","The chemical symbol for gold is","Paris is the capital of","The first president of the United States was George","The Great Wall is in","Toyota is a car company from","Harry Potter was written by J.K.","The Mona Lisa was painted by Leonardo da","Tokyo is the capital of","Rome is the capital of","The Amazon river is in South","Elvis Presley was a famous"]
for q in qs:
    lg=m(**tok(q,return_tensors='pt')).logits[0,-1].softmax(-1); t=lg.topk(3)
    print(repr(q),[(tok.decode([i]),round(v,2)) for v,i in zip(t.values.tolist(),t.indices.tolist())])
