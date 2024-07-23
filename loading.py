import os
from tinygrad.nn.state import safe_load,safe_load_metadata

cwd = os.getcwd()

sf = cwd + "/models/model.safetensors"
sf2   = cwd  + "/models/tinymod.safetensor"
# t,j,tt = safe_load_metadata(sf)
# print(type(t))
# print()
# print(j)
# print()
# print(tt)
weights = safe_load(sf2) 

"""
'model.norm.weight': <Tensor <LB DISK: ./decode-gguf/models/model.safetensors (2048,) 
bfloat16 (<LoadOps.VIEW: 7>, None)> 
on DISK: ./decode-gguf/models/model.safetensors 
with grad None>}
"""