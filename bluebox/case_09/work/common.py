import sys; sys.path.insert(0, "/Users/alex/Documents/alex/weight-space-programming/bluebox")
import tools, torch
tools.DEV = "cpu"
torch.set_grad_enabled(False)
MP = "/Users/alex/Documents/alex/weight-space-programming/bluebox/case_09/model"
def load(): return tools.load_model(MP)
