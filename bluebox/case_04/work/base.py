import sys; sys.path.insert(0,"/Users/alex/Documents/alex/weight-space-programming/bluebox")
import tools, torch
tools.DEV="cpu"; torch.set_grad_enabled(False)
C="/Users/alex/Documents/alex/weight-space-programming/bluebox/case_04"
assert C.endswith("case_04")
m=tools.load_model(C+"/model")
