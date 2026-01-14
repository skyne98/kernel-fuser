# Qwen 3 Numpy Prototype Summary

## Status
**Completed & Verified** (2026-01-14)

## Artifacts
- **Code**: `prototypes/qwen-numpy/qwen.py`
- **Docs**: `prototypes/qwen-numpy/README.md`

## Architecture Details (Critical for Rust Port)
1.  **GGUF Parsing**:
    - Implemented native v3 reader using `mmap`.
    - Key Metadata: `qwen.attention.head_count`, `qwen.rope.freq_base` (1e6), `qwen.attention.layer_norm_rms_epsilon`.
    - **Note**: Qwen 3 uses `Q-Norm` and `K-Norm` (RMSNorm on Q/K heads before RoPE).

2.  **Quantization (Fused)**:
    - **Strategy**: Keep weights in RAM as `QuantizedTensor` (Int8 Quants + F32 Scales).
    - **Compute**: `matmul_fused(x, w)` -> `(x_blocked @ w.q) * w.s`.
    - **Formats**:
        - `Q8_0`: Block 32. Delta F16.
        - `Q8_K`: Block 256. Delta F32.

3.  **RoPE**:
    - Style: **NEOX** (Split Half).
    - Formula: Rotate `(x[i], x[i+half])` not `(x[2i], x[2i+1])`.
    - Base: `1,000,000.0`.

4.  **KV Cache**:
    - Pre-allocated Numpy arrays `[Layer, 2, Seq, Head, Dim]`.
    - Zero-copy slicing for attention history.

5.  **Verification**:
    - Matched `llama.cpp` output for `Qwen3-0.6B-UD-Q8_K_XL.gguf`.
    - Input: "User: Hello!\nAssistant:"
    - Output: " Hello! How can I assist you today?"

## Next Steps
- Port `GGUFReader` logic to Rust (using `memmap2` crate?).
- Implement `dequantize_q8` kernels in CubeCL.
- Implement `matmul_fused` kernel in CubeCL (block-wise accumulation).