from safetensors.torch import load_file
import torch
sd=load_file("/Users/alex/Documents/alex/weight-space-programming/bluebox/case_08/model/model.safetensors")
for k,v in sd.items():
    v=v.float().flatten()
    bf=(v.bfloat16().float()==v).float().mean().item(); hf=(v.half().float()==v).float().mean().item()
    low=(v.view(torch.int32)&0xFF==0).float().mean().item()
    lowbits=(v.view(torch.int32)&0x3).float().mean().item()
    print(f"{k:40s} bf16exact {bf:.3f} fp16exact {hf:.3f} low8zero {low:.3f}")
