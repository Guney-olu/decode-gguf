"""
Loading a gguf file using tinygrad 
It dequantize it and load it 
check https://github.com/tinygrad/tinygrad/blob/master/extra/models/llama.py
for from llama import Transformer, convert_from_huggingface 
"""

import os
from tinygrad.nn.state import safe_load,safe_load_metadata,load_state_dict,get_state_dict
from tinygrad.helpers import getenv
from tinygrad import nn, Variable,Tensor,Device
from llama import Transformer, convert_from_huggingface
from tinygrad.helpers import CI, tqdm
from tinygrad.multi import MultiLazyBuffer
from parser import load_gguf,load_gguf_tensor
from helper import translate_name
from sentencepiece import SentencePieceProcessor
import numpy as np
import time
cwd = os.getcwd()

f2 = cwd + "/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
with open(f2, "rb") as f:
    info , tensor = load_gguf(f)
    tens_dict = {}
    for name in tensor:
        weights = load_gguf_tensor(f, tensor, name)
        shape = tensor[name]["shape"]
        if ".attn_k." in name or ".attn_q." in name:
            num_heads = info["llama.attention.head_count"]
            tmp_shape = (shape[-1] // num_heads // 2, num_heads, 2, shape[0])
            weights = weights.reshape(tmp_shape)
            weights = weights.transpose(0, 2, 1, 3)
            weights = weights.reshape(shape[::-1])
        
        t_tensor_view = translate_name(name)
        # print(f"{t_tensor_view} : {Tensor(weights)}")
        tens_dict[t_tensor_view] = Tensor(weights.astype(np.float16))
        
        
weights = tens_dict 
params = {'dim': 2048, 'n_heads': 32, 'n_kv_heads': 4, 'n_layers': 22, 'norm_eps': 1e-05, 'vocab_size': 32000, 'hidden_dim': 5632}
jit = bool(getenv("JIT", 1))
linear = nn.Linear
model = Transformer(**params, linear=linear, max_context=4096, jit=jit)

if "model.embed_tokens.weight" in weights:
    weights = convert_from_huggingface(weights, model, params["n_heads"], params["n_kv_heads"])

model_state_dict = get_state_dict(model)
load_state_dict(model, weights, strict=False, consume=True)

tokenizer = SentencePieceProcessor(model_file="/models/tokenizer.model")

def Benchmark(model,tokenizer, max_tokens, temperature, device): 
    toks = [tokenizer.bos_id()]
    start_pos = 0
    start_time = time.time()
    for i in range(max_tokens):
        start_pos_var = 0 if start_pos == 0 else Variable("start_pos", 1, 1024).bind(start_pos)
        if isinstance(start_pos_var, Variable):
            start_pos_val = start_pos_var.val 
        else:
            start_pos_val = start_pos_var
        tok_tensor = model(Tensor([toks[start_pos:]], device=device), start_pos_val, temperature)
        tok_tensor.realize()
        tok = tok_tensor.item()

        toks.append(tok)
        start_pos += 1
        decoded_output = tokenizer.decode(toks)
        print(decoded_output)
        current_time = time.time()
        elapsed_time = current_time - start_time
        tokens_per_sec = (i + 1) / elapsed_time
        print(f"Token {i + 1}/{max_tokens}: {decoded_output[-len(tokenizer.decode([tok])):]}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds, Tokens/sec: {tokens_per_sec:.2f}")

    total_time = time.time() - start_time
    print(f"Generated {max_tokens} tokens in {total_time:.2f} seconds ({tokens_per_sec:.2f} tokens/sec)")

    return tokenizer.decode(toks)
shard = 1
device = tuple(f"{Device.DEFAULT}:{i}" for i in range(shard)) if shard > 1 else Device.DEFAULT

print(Benchmark(model,tokenizer,10,0,device))

# verbose = True
# for k,v in (t := tqdm(model_state_dict.items(), disable=CI or not verbose)):
#     print(k,v)
#     if isinstance((mlb:=v.lazydata), MultiLazyBuffer):
#         if isinstance(state_dict[k].lazydata, MultiLazyBuffer): 
#             v.replace(state_dict[k]).realize()
#         else: 
#             v.replace(state_dict[k].shard(mlb.device, mlb.axis)).realize()
#     else: 
#         v.replace(state_dict[k].to(v.device)).realize()
#     consume = True
#     if consume: del state_dict[k]


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


