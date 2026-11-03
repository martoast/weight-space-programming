from common import *
m = load(); tok=tools.tok
W=m.transformer.wte.weight; n=W.norm(dim=1)
for grp in [[" 13"," 14"," 15"," 16"," 17"," 18"],[" A"," B"," C"," D"," E"," F"," G"],[" October"," November"," December"," January"],[" rich"," poor"," high"," low"," hot"," cold"],[" the"," a"," her"," them"]]:
    print([(w, round(n[tok(w).input_ids[0]].item(),2)) for w in grp])
