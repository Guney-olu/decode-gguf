"""
Source -->https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
,https://github.com/99991/pygguf/blob/main/gguf.py
"""
import struct
import warnings

DATA_TYPES = {
    "uint8": 0,
    "int8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "bool": 7,
    "string": 8,
    "array": 9,
    "uint64": 10,
    "int64": 11,
    "float64": 12,
}


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


import os
cwd = os.getcwd()
f2 = cwd + "/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
with open(f2, "rb") as f:
    info , tensor = load_gguf(f)
    print(info.keys())
    print(tensor.keys())    

