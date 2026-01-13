# Qwen Numpy Prototype

This directory contains a **pure Numpy implementation** of the Qwen 2.5/3 transformer architecture.

It serves as a reference implementation for verifying math and tensor operations before porting them to high-performance kernels (like CubeCL or CUDA).

## Status: Verified ✅

This implementation has been **verified to match `llama.cpp` exactly** (token-for-token) when running inference on `Qwen3-0.6B-UD-Q8_K_XL.gguf`.

**Output Match:**
```
User: Hello!
Assistant: Hello! How can I assist you today?
```

## Features

- **Native GGUF Reader**: Implements a complete GGUF v3 parser in pure Python. No external C++ bindings are required to load weights.
- **Quantization Support**: Natively dequantizes the following types into Float32 for Numpy arithmetic:
  - `Q8_0` (Block size 32, f16 delta)
  - `Q8_K` (Block size 256, f32 delta)
  - *F16 and F32 tensors are also supported.*
- **Architecture**:
  - RMSNorm (Pre-Norm & QK-Norm)
  - RoPE (Rotary Positional Embeddings) with high frequency base (1M) and **NEOX-style (split-half) rotation**.
  - Grouped Query Attention (GQA) with broadcasting
  - SwiGLU Feed-Forward Network
- **Inference**:
  - KV Cache implementation.
  - Greedy and Temperature sampling.
  - Integration with `llama-cpp-python` for accurate tokenization.

## Prerequisites

```bash
pip install numpy llama-cpp-python
```

*Note: `llama-cpp-python` is used only for the **Tokenizer** and for running the reference comparison. The model loading, weight dequantization, and inference math are 100% pure Numpy.*

## Usage

1. **Obtain a Model**:
   Place a Qwen GGUF model in the `../../models/` directory.
   Tested with: `Qwen3-0.6B-UD-Q8_K_XL.gguf`.

2. **Run the Script**:

   ```bash
   python qwen.py --prompt "User: Hello!\nAssistant:" --steps 20
   ```

3. **Verify against Reference**:
   Run with `--ref` to run the same prompt through `llama.cpp`'s internal engine and compare outputs side-by-side.

   ```bash
   python qwen.py --ref
   ```

## Implementation Details

### GGUF Parsing
The `GGUFReader` class uses memory mapping (`mmap`) to efficiently parse the binary file format. It extracts metadata (KV pairs) to automatically configure the model dimensions (layers, heads, embedding size, RoPE freq) and locates tensor offsets.

### Dequantization
To perform math in Numpy, quantized weights must be converted to floats.
- **Q8_0**: Reads 34-byte blocks. Extracts a 16-bit float scale and 32 8-bit integers.
- **Q8_K**: Reads 260-byte blocks. Extracts a 32-bit float scale and 256 8-bit integers.

This logic mirrors the C++ implementation in `llama.cpp` `ggml-quants.c`.

### RoPE (Rotary Embeddings)
Qwen uses the "NEOX" style of RoPE, where the rotation is applied to split halves of the head vector:
```python
x1, x2 = x[..., :half], x[..., half:]
x_rotated = cat(x1 * cos - x2 * sin, x1 * sin + x2 * cos)
```
This differs from the "Interleaved" style `(-x[1], x[0], -x[3], x[2]...)` often found in other implementations. Getting this correct was crucial for matching the reference output.

## Goals

1. **Math Verification**: Validate that the manual implementation of RoPE, RMSNorm, and SwiGLU produces correct behavior. **(Completed)**
2. **Kernel Design**: Use this clean Python code as a blueprint for writing the Rust/CubeCL kernels.