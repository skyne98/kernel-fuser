# Qwen3-MoE Implementation Specification

## 1. Overview
This document specifies the architecture and file format for the **Qwen3-MoE** (Mixture-of-Experts) models, specifically targeting the **Qwen3-30B-A3B** variant.

**CRITICAL ARCHITECTURE NOTE:**
Unlike Qwen2-MoE and DeepSeek-V3, the standard Qwen3-MoE architecture (specifically the **30B-A3B** variant) **does NOT use shared experts**. It relies purely on a sparse mixture of routed experts.
- **Shared Experts:** 0
- **Total Experts:** 128
- **Active Experts per Token:** 8

## 2. File Format Structure
The model file is a binary blob with a 256-byte header followed by weights.

### 2.1 Header Layout
The header is 256 bytes total, little-endian. Extensions from the standard Qwen3 header are marked.

| Offset | Type | Description | Value/Note |
| :--- | :--- | :--- | :--- |
| 0x00 | `u32` | Magic Number | `0x6D6F6533` ("moe3") |
| 0x04 | `i32` | Version | `1` |
| 0x08 | `i32` | `dim` | Transformer dimension |
| 0x0C | `i32` | `hidden_dim` | Expert FFN hidden dimension |
| 0x10 | `i32` | `n_layers` | Number of layers |
| 0x14 | `i32` | `n_heads` | Number of query heads |
| 0x18 | `i32` | `n_kv_heads` | Number of KV heads |
| 0x1C | `i32` | `vocab_size` | Vocabulary size |
| 0x20 | `i32` | `max_seq_len` | Max sequence length |
| 0x24 | `i32` | `head_dim` | Dimension of single head |
| 0x28 | `i32` | `shared_classifier` | boolean (0/1) |
| 0x2C | `i32` | `group_size` | Quantization group size |
| 0x30 | `i32` | `num_experts` | Total number of experts (128) |
| 0x34 | `i32` | `num_experts_per_tok` | Active experts per token (8) |
| 0x38 | `i32` | `norm_topk_prob` | boolean (1 = normalize post-topk) |
| 0x3C | `u8` | Padding | Zero-padded to 256 bytes |

### 2.2 Weight Layout
Weights appear in this exact order.

**A. Unquantized FP32 Weights (RMS Norms)**
1. `rms_att_weight`: `[n_layers, dim]`
2. `rms_ffn_weight`: `[n_layers, dim]`
3. `rms_final_weight`: `[dim]`
4. `q_norm_weights`: `[n_layers, head_dim]`
5. `k_norm_weights`: `[n_layers, head_dim]`

**B. Quantized Q8_0 Weights**
1. `token_embedding_table`: `[vocab_size, dim]`
2. **Layer Weights** (Iterate `n_layers` times):
   - **Attention Weights (Standard):**
     - `wq`: `[dim, n_heads * head_dim]`
     - `wk`: `[dim, n_kv_heads * head_dim]`
     - `wv`: `[dim, n_kv_heads * head_dim]`
     - `wo`: `[n_heads * head_dim, dim]`
   - **MoE Weights:**
     - `gate`: `[dim, num_experts]` (The router)
     - `experts_w1`: `[num_experts, dim, hidden_dim]` (Gate)
     - `experts_w2`: `[num_experts, hidden_dim, dim]` (Down)
     - `experts_w3`: `[num_experts, dim, hidden_dim]` (Up)
3. `wcls`: `[dim, vocab_size]` (If not shared)

*Note: In the file, expert weights are flattened. `experts_w1` is stored as `num_experts` contiguous blocks of `[dim, hidden_dim]` weights.*

## 3. MoE Architecture Flow

### 3.1 Forward Pass
```
Input Token -> Embedding -> x
Loop Layers:
  1. Attention Phase (Same as Dense Qwen3):
     - Norm, RoPE, Attention, Residual Connection.
     - x = x + Attention(RMSNorm(x))

  2. MoE Phase:
     - xb = RMSNorm(x)
     
     - **Router (Gating):**
       - logits = MatMul(xb, GateWeights)  // Shape: [num_experts]
       - probs = Softmax(logits)           // Standard Softmax
       - topk_probs, topk_indices = TopK(probs, k=num_experts_per_tok)
       
       - **Normalization:** (If norm_topk_prob == 1)
         - sum_probs = Sum(topk_probs)
         - weights = topk_probs / sum_probs
       - Else:
         - weights = topk_probs

     - **Expert Execution:**
       - final_hidden = 0
       - For i in 0..k-1:
           - idx = topk_indices[i]
           - weight = weights[i]
           - // Expert FFN (SwiGLU)
           - h1 = MatMul(xb, experts_w1[idx])
           - h3 = MatMul(xb, experts_w3[idx])
           - hb = h1 * SiLU(h3)
           - out = MatMul(hb, experts_w2[idx])
           - final_hidden += out * weight
     
     - Residual: x += final_hidden

End Loop
Final Norm: x = RMSNorm(x)
Logits: logits = MatMul(x, Wcls)
```

### 3.2 Key Differences
1.  **No Shared Experts:** Confirmed for Qwen3-30B-A3B. Logic for summing a "shared expert" output is removed.
2.  **Top-K Normalization:** `norm_topk_prob` is typically `true` for Qwen3. The selected top-k probabilities are re-normalized to sum to 1.0 before being used as weights.

## 4. Inference State & Buffers

### 4.1 RunState Additions
To support MoE, the `RunState` struct needs additional buffers:

- `moe_logits`: `[num_experts]` (fp32) - Router output.
- `expert_probs`: `[num_experts_per_tok]` (fp32) - Selected expert weights.
- `expert_indices`: `[num_experts_per_tok]` (int32) - Selected expert IDs.
- `expert_out`: `[dim]` (fp32) - Accumulator for weighted expert outputs.

### 4.2 Compute Optimization
- **Batching:** When running with batch size > 1, sort tokens by expert index to perform batched matrix multiplications for each expert.
- **Quantization:** Expert weights are quantized (Q8_0). Dequantization happens on-the-fly just like the dense layers.

## 5. Implementation Details

### 5.1 Routing Logic
```c
// Pseudo-code for Top-K Softmax Gating with Normalization
void router_forward(float* x, float* gate_w, int* indices, float* weights, int k, int norm_topk) {
    // 1. Compute logits: x @ gate_w
    float logits[NUM_EXPERTS];
    matmul(logits, x, gate_w, ...);

    // 2. Softmax over ALL experts first
    softmax(logits, NUM_EXPERTS);

    // 3. Top-K Selection
    // Select top k probabilities and their indices
    top_k_select(logits, indices, weights, k);

    // 4. Re-normalization (Standard in Qwen3)
    if (norm_topk) {
        float sum = 0;
        for(int i=0; i<k; i++) sum += weights[i];
        for(int i=0; i<k; i++) weights[i] /= sum;
    }
}
```

### 5.2 Expert Execution Loop
Iterating through selected experts:
```c
// Clear accumulator
memset(final_output, 0, dim * sizeof(float));

for (int i = 0; i < k; i++) {
    int ex_idx = indices[i];
    float weight = weights[i];

    // Select specific expert weights
    QuantizedTensor* w1 = &experts_w1[ex_idx];
    QuantizedTensor* w2 = &experts_w2[ex_idx];
    QuantizedTensor* w3 = &experts_w3[ex_idx];

    // Run FFN
    ffn_forward(xb, w1, w2, w3, expert_output_buffer);

    // Accumulate weighted output
    for(int j=0; j<dim; j++) {
        final_output[j] += expert_output_buffer[j] * weight;
    }
}
```