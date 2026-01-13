# Qwen3.c Implementation Specification

## 1. File Format Structure
The model file is a binary blob with a 256-byte header followed by weights.

### 1.1 Header Layout
The header is 256 bytes total, little-endian.

| Offset | Type | Description | Value/Note |
| :--- | :--- | :--- | :--- |
| 0x00 | `u32` | Magic Number | `0x616A6331` ("ajc1") |
| 0x04 | `i32` | Version | `1` |
| 0x08 | `i32` | `dim` | Transformer dimension |
| 0x0C | `i32` | `hidden_dim` | FFN hidden dimension |
| 0x10 | `i32` | `n_layers` | Number of layers |
| 0x14 | `i32` | `n_heads` | Number of query heads |
| 0x18 | `i32` | `n_kv_heads` | Number of KV heads |
| 0x1C | `i32` | `vocab_size` | Vocabulary size |
| 0x20 | `i32` | `max_seq_len` | Max sequence length |
| 0x24 | `i32` | `head_dim` | Dimension of single head |
| 0x28 | `i32` | `shared_classifier` | boolean (0/1) |
| 0x2C | `i32` | `group_size` | Quantization group size |
| 0x30 | `u8` | Padding | Zero-padded to 256 bytes |

### 1.2 Weight Layout
Weights appear in this exact order.

**A. Unquantized FP32 Weights (RMS Norms)**
1. `rms_att_weight`: `[n_layers, dim]`
2. `rms_ffn_weight`: `[n_layers, dim]`
3. `rms_final_weight`: `[dim]`
4. `q_norm_weights`: `[n_layers, head_dim]` (Qwen3 specific: Q-norm)
5. `k_norm_weights`: `[n_layers, head_dim]` (Qwen3 specific: K-norm)

**B. Quantized Q8_0 Weights**
Each quantized tensor consists of two parts stored consecutively:
1. `q`: `int8` tensor of size `[N]`
2. `s`: `float32` scales of size `[N / group_size]`

Order of quantized blocks:
1. `token_embedding_table`: `[vocab_size, dim]`
2. Layer Weights (each `[n_layers, ...]`):
   - `wq`: `[dim, n_heads * head_dim]`
   - `wk`: `[dim, n_kv_heads * head_dim]`
   - `wv`: `[dim, n_kv_heads * head_dim]`
   - `wo`: `[n_heads * head_dim, dim]`
   - `w1`: `[dim, hidden_dim]` (Gate)
   - `w2`: `[hidden_dim, dim]` (Down)
   - `w3`: `[dim, hidden_dim]` (Up)
3. `wcls`: `[dim, vocab_size]` (Only if `shared_classifier == 0`)

## 2. Transformer Architecture

### 2.1 Forward Pass Data Flow
```
Input Token -> Embedding -> x
Loop Layers:
  1. Attention Norm: xb = RMSNorm(x)
  2. QKV Proj: q, k, v = MatMul(xb, Wq), MatMul(xb, Wk), MatMul(xb, Wv)
  3. QK Norm (Qwen3): q = RMSNorm(q), k = RMSNorm(k)
  4. RoPE: Rotate q, k
  5. Attention:
     - Update KV Cache
     - Scores = (q . k^T) / sqrt(head_dim)
     - Softmax(Scores)
     - xb = Scores . v
  6. Output Proj: xb = MatMul(xb, Wo)
  7. Residual: x += xb
  8. FFN Norm: xb = RMSNorm(x)
  9. FFN (SwiGLU):
     - h1 = MatMul(xb, W1)
     - h3 = MatMul(xb, W3)
     - hb = h1 * SiLU(h3)  <-- NOTE: Implementation uses w1(x) * w3(x) * sigmoid(w1(x)) ??
         * Check C source:
           s->hb[i] *= s->hb2[i] * (1.0f / (1.0f + expf(-s->hb[i])));
           Wait, logic is: w1(x) * w3(x) * sigmoid(w1(x)) ??
           Correct Qwen/Llama SwiGLU is (SiLU(Gate) * Up).
           C Code:
             matmul(s->hb, ..., w1); // Gate?
             matmul(s->hb2, ..., w3); // Up?
             s->hb[i] *= s->hb2[i] * sigmoid(s->hb[i])
           This implies `hb` is the gate, and we apply SiLU to it.
  10. Down Proj: xb = MatMul(hb, W2)
  11. Residual: x += xb
End Loop
Final Norm: x = RMSNorm(x)
Logits: logits = MatMul(x, Wcls)
```

### 2.2 Quantization Scheme (Q8_0)
Block-wise symmetric quantization.
- Block size: `group_size` (typically 64, 128)
- Formula: `real_value = int8_value * scale_factor`
- Max int8 value 127 maps to max absolute value in block.

## 3. Tokenizer Format (.tokenizer)
Binary file accompanying the model.

| Offset | Type | Description |
| :--- | :--- | :--- |
| 0x00 | `u32` | `max_token_length` |
| 0x04 | `u32` | `bos_token_id` |
| 0x08 | `u32` | `eos_token_id` |
| ... | ... | **Loop `vocab_size` times:** |
| | `f32` | Merge score |
| | `u32` | Token string length (`len`) |
| | `u8` * `len` | UTF-8 Token bytes |

## 4. Inference State
### 4.1 KV Cache
- **Key Cache**: `[n_layers, seq_len, n_kv_heads, head_dim]`
- **Value Cache**: `[n_layers, seq_len, n_kv_heads, head_dim]`
- Type: `fp32` (in C implementation).

### 4.2 RunState Buffers
- `x`, `xb`: `dim` (fp32)
- `hb`, `hb2`: `hidden_dim` (fp32)
- `q`: `n_heads * head_dim` (fp32)
- `k`, `v`: `n_kv_heads * head_dim` (fp32)
- `logits`: `vocab_size` (fp32)
- Quantized intermediate buffers (`xq`, `hq`) for matrix multiplication inputs.

## 5. Critical Implementation Details
1. **RoPE**: Applied after QK-Norm.
   - Frequency base: `1,000,000` (Qwen default) vs `10,000` (Llama).
   - Only applies to the first half of head_dim? (C code: `j < p->head_dim/2` and operates on pairs `q[j]` and `q[j + head_dim/2]`).
   - This suggests full rotary embedding (since it pairs `i` with `i + dim/2`).

2. **QKV Packing**:
   - C implementation stores weights as separate tensors `wq`, `wk`, `wv`.
   - `n_kv_heads` can be smaller than `n_heads` (GQA).

3. **MatMul Kernel**:
   - `int8` activation * `int8` weight.
   - Accumulate in `i32`, convert to `f32`, multiply by `scale_activation * scale_weight`.
   - C implementation quantizes activations on the fly before every linear layer.

```c
// Dequantize logic for reference
val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
```

## 6. Rust Implementation Goals (Kernel-Fuser)
1. **Loader**: `mmap` the binary file, parse header, create structs for weights.
2. **Tokenizer**: Read `.tokenizer` file, implement BPE encode/decode.
3. **Inference**:
   - Implement `forward()` in pure Rust first (CPU).
   - Port heavy ops (`MatMul`, `RMSNorm`, `RoPE`) to **CubeCL**.
4. **Sampler**: Argmax, Top-P, Temperature.
