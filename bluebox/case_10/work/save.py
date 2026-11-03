from base import *
from fix import apply_fix
C="/Users/alex/Documents/alex/weight-space-programming/bluebox/case_10"
m = apply_fix(tools.load_model(MP))
print(tools.save_fix(m, C+"/model", C+"/fix_patch.pt"))
