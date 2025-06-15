#!/usr/bin/env python3

# call this way
# python inspect.py llama3_3b_f32.gguf | head -n 20
# inspect metadata after generating gguf files.
import struct, sys, pathlib

def le32(b):  return struct.unpack("<I", b)[0]
def le64(b):  return struct.unpack("<Q", b)[0]

def next_string(buf, pos):
    n = le64(buf[pos:pos+8]); pos += 8
    s = buf[pos:pos+n];       pos += n
    return s.decode("utf-8", "ignore"), pos

gguf = pathlib.Path(sys.argv[1]).read_bytes()

magic = gguf[:4]; version = le32(gguf[4:8])
t_count  = le64(gguf[8:16]); kv_count = le64(gguf[16:24])
print(f"magic={magic}  version={version}  kv_count={kv_count}")

pos = 24
for i in range(kv_count):
    key, pos = next_string(gguf, pos)
    vtype = le32(gguf[pos:pos+4]); pos += 4
    print(f"{i:02} {key:<30} vtype={vtype}")
    if vtype == 8:                 # STRING
        ln = le64(gguf[pos:pos+8]); pos += 8 + ln
    elif vtype == 4:               # UINT32
        pos += 4
    elif vtype == 6:               # FLOAT32
        pos += 4
    elif vtype == 10:              # UINT64
        pos += 8
    else:                          # any other: skip with len header
        elem = le32(gguf[pos:pos+4]); pos += 4
        ln   = le64(gguf[pos:pos+8]); pos += 8 + ln
