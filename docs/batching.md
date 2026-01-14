# Deep Dive: Batching for Prompt Processing (PP) and Token Generation (TG)

This document provides a comprehensive architectural analysis of batching strategies in Large Language Model (LLM) inference. It covers the mathematical motivations (arithmetic intensity), scheduling algorithms (continuous batching), memory management (PagedAttention), and low-level implementation details relevant to high-performance engines like `llama.cpp`, `vLLM`, and custom Rust/CubeCL implementations.

---

## 1. Theory: The Arithmetic Intensity Gap

To understand why batching is critical, we must analyze the **Arithmetic Intensity (AI)** of the workload. AI is the ratio of floating-point operations (FLOPs) performed per byte of memory accessed ($AI = \frac{\text{FLOPs}}{\text{Bytes}}$).

Modern hardware (e.g., NVIDIA A100) has a massive gap between compute capability and memory bandwidth:
*   **Peak Compute (FP16 Tensor):** ~312 TFLOPS
*   **Memory Bandwidth (HBM2e):** ~2,039 GB/s
*   **Hardware "Roofline" Ratio:** $\approx 153$ FLOPs/Byte.

**Implication:** To fully utilize the GPU's compute cores, an algorithm must perform ~153 operations for every byte loaded.

### 1.1 The Math of Token Generation (Batch Size = 1)
For a model with $P$ parameters (e.g., Llama-3-70B, where $P \approx 70 \times 10^9$):
1.  **Memory Access:** We must load all weights $W$ once. Assuming FP16 (2 bytes), traffic is $2P$ bytes.
2.  **Compute:** Matrix-Vector multiplication. For each weight, we do roughly 1 multiply-add (2 FLOPs). Total FLOPs $\approx 2P$.
3.  **Intensity:**
    $$ AI_{TG1} = \frac{2P \text{ FLOPs}}{2P \text{ Bytes}} = 1 \text{ FLOP/Byte} $$

**Result:** With an AI of 1, we are orders of magnitude below the hardware's roofline (153). The GPU cores sit idle 99% of the time, waiting for memory. This is **Memory Bound**.

### 1.2 The Math of Batched Generation (Batch Size = $B$)
If we generate tokens for $B$ users simultaneously:
1.  **Memory Access:** Weights are loaded **once** ($2P$ bytes) and reused across all $B$ vectors in the batch.
2.  **Compute:** Matrix-Matrix multiplication. We perform calculations for $B$ tokens. Total FLOPs $\approx 2P \times B$.
3.  **Intensity:**
    $$ AI_{TGB} = \frac{2P \cdot B \text{ FLOPs}}{2P \text{ Bytes}} = B \text{ FLOPs/Byte} $$

**Result:** If $B=128$, arithmetic intensity rises to 128, approaching the hardware roofline. We get ~128x more throughput for nearly the same memory cost.

---

## 2. Workload Profiles: PP vs. TG

LLM inference is a hybrid workload requiring distinct handling for its two phases.

### Prompt Processing (PP) - "Prefill"
*   **Operation:** $Q \times K^T \times V$ where Sequence Length $L > 1$.
*   **Kernel:** **GEMM** (General Matrix-Matrix Multiplication).
*   **Nature:** **Compute-bound**. Even with Batch Size=1, a long prompt ($L=2048$) acts like a "batch" of 2048 tokens.
*   **Bottleneck:** TFLOPS.
*   **Latency:** Time To First Token (TTFT).

### Token Generation (TG) - "Decoding"
*   **Operation:** Auto-regressive generation where $L_{new} = 1$.
*   **Kernel:** **GEMV** (General Matrix-Vector Multiplication) / Small-N GEMM.
*   **Nature:** **Memory-bound**. Requires batching to saturate bandwidth.
*   **Bottleneck:** HBM Bandwidth.
*   **Latency:** Time Between Tokens (TBT).

---

## 3. Evolution of Batching Strategies

### 3.1 Static Batching
The "naive" approach.
1.  Wait for $N$ requests to arrive.
2.  Pad all requests to the length of the longest prompt in the batch.
3.  Process until *all* requests generate an `<EOS>` token.

**The "Bubble" Problem (Head-of-Line Blocking):**
If Request A needs 10 tokens and Request B needs 1000 tokens:
*   Steps 1-10: GPU processes A and B.
*   Steps 11-1000: GPU processes B, while A's slot contains padding (waste).
*   **Efficiency Loss:** Massive. 50%+ of compute can be wasted on padding.

### 3.2 Continuous Batching (Iteration-Level Scheduling)
Introduced by Orca (OSDI '22), popularized by vLLM and TGI.
**Core Idea:** The batch size is dynamic per *iteration* (forward pass), not per *request group*.

**Scheduler Pseudocode:**
```python
queue = RequestQueue()
running_sequences = []

while True:
    # 1. RETIRE: Remove completed sequences immediately
    active_seqs = []
    for seq in running_sequences:
        if seq.is_finished():
            kv_cache.free(seq.id)
        else:
            active_seqs.append(seq)
    
    # 2. SCHEDULE: Fill remaining slots with new requests (Prefill)
    #    Prioritize keeping the "Token Budget" full
    while queue.has_pending() and kv_cache.has_space():
        new_req = queue.pop()
        kv_cache.allocate(new_req)
        active_seqs.append(new_req)
        
    # 3. EXECUTE: Run one forward pass
    #    This batch contains a mix of:
    #    - 1st token generation (Prefill) for new_req
    #    - Nth token generation (Decode) for existing seqs
    logits = model.forward(active_seqs)
    
    # 4. DECODE: Sample next tokens
    for seq, seq_logits in zip(active_seqs, logits):
        next_token = sample(seq_logits)
        seq.append(next_token)
```

### 3.3 Chunked Prefill (uBatching)
Used in `llama.cpp` (`n_ubatch`).
**The Conflict:** A new request with 8k tokens takes significantly longer to prefill (e.g., 500ms) than a decode step (e.g., 20ms).
**Symptom:** Existing users see their generation "stutter" every time a new long prompt enters the system.
**Solution:**
1.  Limit the number of "prefill tokens" per iteration (e.g., 512).
2.  Split the 8k prompt into 16 chunks.
3.  Interleave:
    *   Iter 1: Prefill Chunk 1 (512 tokens) + Decode Users B, C, D (1 token each).
    *   Iter 2: Prefill Chunk 2 (512 tokens) + Decode Users B, C, D (1 token each).
    *   ...
4.  **Result:** TBT remains consistent (~30ms) instead of spiking to 500ms.

---

## 4. Memory Management: The PagedAttention Revolution

Continuous batching creates a memory fragmentation nightmare. If we pre-allocate contiguous VRAM for `max_context_len` (e.g., 8192) for every user, we run out of memory instantly. If we allocate dynamically, we get memory holes.

### 4.1 The Solution: Virtual Memory for KV Cache
**PagedAttention** (vLLM) applies OS virtual memory concepts to the KV cache.

1.  **Blocks:** Divide KV cache into fixed-size blocks (e.g., 16 or 32 tokens).
2.  **Physical Memory:** A large, pre-allocated tensor `[Num_Physical_Blocks, Block_Size, Head_Dim]`.
3.  **Block Table:** A mapping per sequence: `Seq_A -> [Physical_Block_4, Physical_Block_9, Physical_Block_2]`.

### 4.2 Advantages
*   **Zero External Fragmentation:** We only allocate small blocks as needed.
*   **Memory Sharing (Copy-on-Write):** Critical for advanced decoding (Beam Search, Parallel Sampling).
    *   Prompt: "Write a poem."
    *   Beam 1 and Beam 2 start with the *same* KV blocks for the prompt.
    *   They only allocate *new* blocks when their generated tokens diverge.
    *   Reduces memory usage by 50-70% for complex sampling.

---

## 5. Implementation Mechanics (Low Level)

### 5.1 The `llama_batch` Structure
`llama.cpp` uses a flattened structure to represent a heterogeneous batch.

```cpp
struct llama_batch {
    int32_t n_tokens;       // Total tokens in this forward pass
    
    // Arrays of size [n_tokens]
    llama_token * token;    // The token IDs
    llama_pos   * pos;      // Absolute position (0, 1, 2... 100... 101)
    
    // Sequence mapping
    int32_t     * n_seq_id; // How many sequences does this token belong to?
    llama_seq_id ** seq_id; // The IDs (e.g., [[0], [1], [1, 2]])
    
    // Output control
    int8_t      * logits;   // 1 = compute logits, 0 = skip
};
```

*   **`pos` vs Index:** The index in the `token` array is just the batch index ($0..B$). The `pos` array tells the RoPE kernel the *semantic* position of that token in its sentence.
*   **`n_seq_id` > 1:** Used when a single token is shared by multiple beams (e.g., a common prefix in beam search).

### 5.2 Logits Masking
In a chunked prefill of 512 tokens, we only care about predicting the *next* token (after token 511).
*   Calculating logits involves `Norm -> Dense -> Softmax` over the vocab size (e.g., 128k).
*   This is expensive!
*   `llama.cpp` sets `logits[0..510] = 0` and `logits[511] = 1`.
*   The final projection kernel skips rows where `logits == 0`, saving significant compute and P2P bandwidth.

### 5.3 Unified KV Cache Layout
Unlike standard Transformers which might use `[Batch, Head, Seq, Dim]`, high-performance engines use layouts optimized for block access:

**Shape:** `[Layer, Physical_Block_Index, Head, Block_Size, Head_Dim]`

*   **Layer:** Outermost dimension to allow layer-wise processing.
*   **Block_Size:** (e.g., 16) Inner dimension for vectorization (SIMD) efficiency during Attention.
*   **Block_Index:** The "address" fetched from the Block Table.

---

## 6. Hardware Impact Summary

| Metric | PP (Prefill) | TG (Decode) | Batched TG |
| :--- | :--- | :--- | :--- |
| **Dominant Resource** | GPU Cores (ALU) | Memory Bandwidth | Memory Bandwidth |
| **Arithmetic Intensity** | High (~Batch size equivalent) | Low (1) | Medium ($B$) |
| **Typical Efficiency** | 60-80% of Peak TFLOPS | 1-5% of Peak TFLOPS | 40-70% of Peak TFLOPS |
| **Latency Metric** | TTFT (Time to First Token) | TBT (Time Between Tokens) | Throughput / Latency Balance |
| **Kernel Type** | Dense GEMM | GEMV / Attention | Batched GEMV |

---

## 7. Key Research Bibliography

1.  **Yu et al. (2022)**. *Orca: A Distributed Serving System for Transformer-Based Generative Models*. OSDI. (The foundation of Continuous Batching).
2.  **Kwon et al. (2023)**. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP. (The vLLM paper).
3.  **Agrawal et al. (2024)**. *Sarathi-Serve: Optimal Scheduling for LLM Inference*. (Chunked prefill and TBT optimization).
4.  **Gerganov et al.** *llama.cpp Source & Discussions*. GitHub. (Implementation of `n_ubatch` and unified KV cache).
5.  **NVIDIA**. *Triton Inference Server Documentation*. (In-flight batching concepts).
6.  **Aminabadi et al. (2022)**. *DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale*. (Early work on hybrid batching).