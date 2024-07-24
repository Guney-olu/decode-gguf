import os
from tinygrad.nn.state import safe_load,safe_load_metadata,load_state_dict,get_state_dict
from tinygrad.helpers import getenv
from tinygrad import nn
from llama import Transformer, convert_from_huggingface
from tinygrad.helpers import prod, argsort, DEBUG, Timing, CI, unwrap, GlobalCounters, tqdm
from tinygrad.multi import MultiLazyBuffer

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
 
params = {'dim': 2048, 'n_heads': 32, 'n_kv_heads': 4, 'n_layers': 22, 'norm_eps': 1e-05, 'vocab_size': 32000, 'hidden_dim': 5632}
jit = bool(getenv("JIT", 1))
linear = nn.Linear
model = Transformer(**params, linear=linear, max_context=4096, jit=jit)

if "model.embed_tokens.weight" in weights:
    weights = convert_from_huggingface(weights, model, params["n_heads"], params["n_kv_heads"])
    print(len(weights))

model_state_dict = get_state_dict(model)
state_dict = weights
verbose = True
for k,v in (t := tqdm(model_state_dict.items(), disable=CI or not verbose)):
    print(k,v)
    if isinstance((mlb:=v.lazydata), MultiLazyBuffer):
        if isinstance(state_dict[k].lazydata, MultiLazyBuffer): 
            v.replace(state_dict[k]).realize()
        else: 
            v.replace(state_dict[k].shard(mlb.device, mlb.axis)).realize()
    else: 
        v.replace(state_dict[k].to(v.device)).realize()
    # consume = True
    # if consume: del state_dict[k]


        


#load_state_dict(model, weights, strict=False, consume=True)


"""
'model.norm.weight': <Tensor <LB DISK: ./decode-gguf/models/model.safetensors (2048,) 
bfloat16 (<LoadOps.VIEW: 7>, None)> 
on DISK: ./decode-gguf/models/model.safetensors 
with grad None>}
"""

"""
 <Tensor <LB METAL (256, 2048) half ShapeTracker(views=(View(shape=(256, 2048), strides=(2048, 1), offset=0, mask=None, contiguous=True),))> 
 on METAL with grad None>, 
 <Tensor <LB METAL (2048,) 
 half (<LoadOps.COPY: 3>, None)> on METAL with grad None>])
"""
"""
<Tensor <LB METAL (1, 8192, 1, 32, 2) half ShapeTracker(views=(View(shape=(1, 8192, 1, 32, 2), 
strides=(0, 64, 0, 2, 1), offset=0, mask=None, contiguous=True),))> 
on METAL with grad None>])

"""


