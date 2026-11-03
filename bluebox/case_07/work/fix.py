from common import *
m = load(); P0=m.transformer.wpe.weight.clone()
m.transformer.wpe.weight[1]=P0[2].clone()
print(tools.skill_report(m), tools.perplexity(m))
print(tools.save_fix(m, MP, "/Users/alex/Documents/alex/weight-space-programming/bluebox/case_07/fix_patch.pt"))
