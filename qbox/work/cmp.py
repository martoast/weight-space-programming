import json,sys,glob,os
def load(tag,task,split):
    d={}
    for f in glob.glob(f'res/{tag}_{task}_{split}_*.json'):
        r=json.load(open(f)); d.update(dict(zip(r['idx'],r['scores'])))
    return d
a=load(sys.argv[1],sys.argv[3],sys.argv[4]); b=load(sys.argv[2],sys.argv[3],sys.argv[4])
k=sorted(set(a)&set(b))
print(f"n={len(k)} {sys.argv[1]}={sum(a[i] for i in k)} {sys.argv[2]}={sum(b[i] for i in k)} gained={sum(b[i]>a[i] for i in k)} lost={sum(b[i]<a[i] for i in k)}")
