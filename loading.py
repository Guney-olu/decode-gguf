import os
from tinygrad.nn.state import safe_load
cwd = os.getcwd()

sf = cwd + "/models/model.safetensors"

weights = safe_load(sf) 
print(weights)

"""
'model.norm.weight': <Tensor <LB DISK: ./decode-gguf/models/model.safetensors (2048,) 
bfloat16 (<LoadOps.VIEW: 7>, None)> 
on DISK: ./decode-gguf/models/model.safetensors 
with grad None>}
"""