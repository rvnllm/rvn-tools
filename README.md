![Build Status](https://github.com/rvnllm/rvn-convert/workflows/rvn-convert%20CI/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![Reddit](https://img.shields.io/badge/Reddit-Join%20Discussion-orange?logo=reddit)](https://www.reddit.com/r/LocalLLaMA/comments/1l8wty9/tool_rvnconvert_oss_rustbased_safetensors_to_gguf/)
[![Built with Rust](https://img.shields.io/badge/built%20with-rust-orange?logo=rust)](https://www.rust-lang.org/)

# rvnllm

**High-Performance Tooling for LLM Inspection, Validation, and Format Conversion**

`rvnllm` is a Rust-based toolkit for analyzing, validating, and converting large language models.  
It enables fast, zero-copy access to model internals for inspection and diagnostics — without requiring inference execution.


## Features

- Convert models between SafeTensors and GGUF formats
- Inspect tensor metadata, shapes, alignment, and memory layout
- Analyze prompt-driven attention and entropy behavior (trace mode)
- Detect malformed tensors, misaligned weights, or corrupted embeddings
- Load multi-gigabyte models in milliseconds using zero-copy design
- Plugin-based parser architecture with extensible format support


## Command Line Tools

CLI Tools

| Tool         | Description                             | Docs |
|--------------|-----------------------------------------|------|
| [`rvn-info`](tools/rvn-info/README.md)     | Display GGUF/SafeTensors metadata and tensor layout | [docs](tools/rvn-info) |
| [`rvn-convert`](tools/rvn-convert/README.md) | Convert between model formats (GGUF, SafeTensors) | [docs](tools/rvn-convert) |
| [`rvn-diff`](tools/rvn-diff/README.md)     | Compare models at the tensor level | [docs](tools/rvn-diff) |
| [`rvn-inspect`](tools/rvn-inspect/README.md) | Perform structural and diagnostic checks | [docs](tools/rvn-inspect) |
| [`rvn-prompt`](tools/rvn-prompt/README.md) | Analyze model behavior in response to a prompt | [docs](tools/rvn-prompt) |

## Python API (Preview)

```python
import rvnllm as rvn

model = rvn.load("model.gguf")

# Access model tensors and metadata
df = model.tensors()           # Returns Polars DataFrame
t = model.tensor("token_embd.weight")

# Diagnostics
t.entropy()
t.norm()

# Trace prompt behavior
p = rvn.prompt("The sky is")
p.inspect()  # Returns diagnostic trace object
```
Supports integration with polars, pandas, numpy, or direct memory access.
Note: This library **does not support inference or sampling.**

## Format Support
| Format      | Load    | Inspect | Convert | Status  |
| ----------- | ------- | ------- | ------- | ------- |
| GGUF        | Completed | Planned | Test  | Beta  |
| SafeTensors | Planned | Planned | Planned | Not yet    |
| PyTorch     | Planned | Planned | Planned | Not yet |
| ONNX        | Planned | Planned | Planned | Not yet |


## Design Goals

- Minimal to zero runtime overhead
- Designed for validation, analysis, and debugging workflows
- Execution/runtime tracing lives in separate layers (non-OSS)
- Modular parser system supports future formats and backends
- Shared core for both CLI and Python interface

## Why Use rvnllm?
Model files are often large, opaque, and difficult to verify.
rvnllm provides fast, format-agnostic access to the internal structure of language models — enabling safer usage, better diagnostics, and improved tooling pipelines.

This is not an inference engine.
This is the layer you use before you trust a model with your input

## License
**MIT License.**
Core functionality is open source and intended for non-commercial use cases.  
Inference execution, advanced runtime tracing, and proprietary format extensions are not included.


## Project
Maintained by **@rvnllm**  
Part of the broader RVN initiative on secure and efficient LLM infrastructure.
