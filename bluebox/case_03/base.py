import sys, os; sys.path.insert(0, "/Users/alex/Documents/alex/weight-space-programming/bluebox")
import tools
if os.environ.get("CPU"): tools.DEV="cpu"
from tools import *
DEV=tools.DEV
torch.set_num_threads(4)
M="/Users/alex/Documents/alex/weight-space-programming/bluebox/case_03/model"
