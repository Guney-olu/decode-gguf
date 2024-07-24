"""
Source -> https://github.com/99991/pygguf/blob/main/gguf.py
"""
import struct
import numpy as np
from helper import DATA_TYPES,GGML_BLOCK_SIZES,GGML_ELEMENTS_PER_BLOCK,GGML_NAMES,GGML_DEQUANTIZE,translate_name


def read_value(f, data_type):
    if data_type == DATA_TYPES["string"]:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    elif data_type == DATA_TYPES["uint32"]:
        return struct.unpack("<I", f.read(4))[0]

    elif data_type == DATA_TYPES["uint64"]:
        return struct.unpack("<Q", f.read(8))[0]

    elif data_type == DATA_TYPES["int64"]:
        return struct.unpack("<q", f.read(8))[0]

    elif data_type == DATA_TYPES["int32"]:
        return struct.unpack("<i", f.read(4))[0]

    elif data_type == DATA_TYPES["float32"]:
        return struct.unpack("<f", f.read(4))[0]

    elif data_type == DATA_TYPES["float64"]:
        return struct.unpack("<d", f.read(4))[0]

    elif data_type == DATA_TYPES["bool"]:
        return struct.unpack("<?", f.read(1))[0]

    elif data_type == DATA_TYPES["uint8"]:
        return struct.unpack("<B", f.read(1))[0]

    elif data_type == DATA_TYPES["int8"]:
        return struct.unpack("<b", f.read(1))[0]

    elif data_type == DATA_TYPES["uint16"]:
        return struct.unpack("<H", f.read(2))[0]

    elif data_type == DATA_TYPES["int16"]:
        return struct.unpack("<h", f.read(2))[0]

    elif data_type == DATA_TYPES["array"]:
        data_type, count = struct.unpack("<IQ", f.read(4 + 8))
        return [read_value(f, data_type) for _ in range(count)]

    else:
        raise NotImplementedError(f"Data type {data_type} not implemented")

def load_gguf(f):
    f.seek(4) #file pointer to position correctly
    values = struct.unpack("<IQQ", f.read(4+8+8))
    _, n_tensors, n_kv = values
    info = {}
    for _ in range(n_kv):
        name = read_value(f, DATA_TYPES["string"])

        data_type = struct.unpack("<I", f.read(4))[0]

        info[name] = read_value(f, data_type)

    tensor = {}
    for _ in range(n_tensors):
        name = read_value(f,DATA_TYPES["string"])
        shape_len = read_value(f,DATA_TYPES["uint32"])
        shape = [read_value(f, DATA_TYPES["uint64"]) for _ in range(shape_len)]
        ggml_type = read_value(f, DATA_TYPES["uint32"])
        bad_offset = read_value(f, DATA_TYPES["uint64"])
        tensor[name] = {
        "ggml_type": ggml_type,
         "shape": shape,
        "bad_offset": bad_offset,
        }
    
    start = f.tell()
    for t in tensor.values():
        offset = start + t["bad_offset"]
        # Alignment is 32 by default.
        # https://github.com/ggerganov/ggml/blob/e1daebbf9d38d510ba456c4d50b4500a73ac2b14/docs/gguf.md?plain=1#L253
        alignment = info.get("general.alignment", 32)
        offset += (alignment - offset % alignment) % alignment

        t["offset"] = offset

    return info, tensor

def load_gguf_tensor(f,tensorinfo,name):
    t = tensorinfo[name]
    offset = t["offset"]
    shape = t["shape"]
    ggml_type = t["ggml_type"]
    
    if ggml_type not in GGML_NAMES:
        raise NotImplementedError(f"ggml_type {ggml_type} not implemented")
    
    ggml_name = GGML_NAMES[ggml_type]
    
    block_size = GGML_BLOCK_SIZES[ggml_name]
    elements_per_block = GGML_ELEMENTS_PER_BLOCK[ggml_name]
    dequantize = GGML_DEQUANTIZE[ggml_name]

    num_elements = np.prod(shape)

    f.seek(offset)

    size = num_elements * block_size // elements_per_block
    data = f.read(size)
    values = dequantize(data)

    return values.reshape(shape[::-1])


"""
Some fails
##--/////----/-//////////-----------//////////////--------//////-
# from tinygrad.dtype import dtypes
# import json

# def hack_safe_load_metadata(tens_dict):
#     for key,val in tens_dict.items():
#         t = val if isinstance(val, Tensor) else Tensor.empty(os.stat(val).st_size, dtype=dtypes.uint8, device=f"disk:{val}")
#         print(t)
        # json_len = t[0:8].bitcast(dtypes.int64).item()
        # metadata = json.loads(t[8:8+json_len].numpy().tobytes())
        # return t,json_len,metadata
    
# hack_safe_load_metadata(tens_dict)

# safe_dtypes = {"BOOL":dtypes.bool, "I8":dtypes.int8, "U8":dtypes.uint8, "I16":dtypes.int16, "U16":dtypes.uint16, "I32":dtypes.int, "U32":dtypes.uint,
#                "I64":dtypes.int64, "U64":dtypes.uint64, "F16":dtypes.float16, "BF16":dtypes.bfloat16, "F32":dtypes.float32, "F64":dtypes.float64}

# def hack_safe_load(tens_dict):
#     t, json_len, metadata = safe_load_metadata(tens_dict)
#     ret = {}
#     for k,v in metadata.items():
#         if k == "__metadata__": continue
#         dtype = safe_dtypes[v['dtype']]
#         sz = (v['data_offsets'][1]-v['data_offsets'][0])
#         ret[k] = t[8+json_len+v['data_offsets'][0]:8+json_len+v['data_offsets'][0]+sz].bitcast(dtype).reshape(v['shape'])
#     return ret

# ret = hack_safe_load(tens_dict)
# print(ret)

 
# from safetensors.torch import load_file
# from safetensors import safe_open
# sf = cwd + "/models/model.safetensors"

# tensors = {}
# with safe_open(sf, framework="pt", device="cpu") as f:
#     for key in f.keys():
#         tensors[key] = f.get_tensor(key)
#     print(tensors)

# state_dict = load_file(sf)
# print(state_dict)
# for key, value in state_dict.items():
#     print(f"{key} {value}")
"""