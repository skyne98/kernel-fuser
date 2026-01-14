# Eagle3 Speculative Decoding Implementation Plan

## 1. Concept Overview

**Eagle3** (Extrapolation Algorithm for Greater Language-model Efficiency, v3) is an advanced speculative decoding technique. Unlike standard speculative decoding which uses a completely separate small model (e.g., Qwen-1.8B drafting for Qwen-72B), Eagle3 typically employs a lightweight **Draft Head** (or "Eagle Head") that sits on top of the frozen target model's features.

### Core Mechanism
1.  **Drafting:** The Eagle head takes features from the target model (embedding, middle layers, final layers) and autoregressively generates a **tree** of candidate tokens.
2.  **Verification:** The target LLM performs a single forward pass on this tree of candidates using **Tree Attention** (a specific attention mask).
3.  **Acceptance:** The target model's logits are compared against the draft tokens. Valid paths in the tree are accepted, and the longest valid prefix is kept.

## 2. Architecture

### 2.1 The "Eagle Head" (Draft Model)
- **Structure:** A small transformer decoder (typically 1-4 layers).
- **Inputs:**
    - The current token embedding.
    - **Feature Fusion:** A concatenated vector of features from the target model's previous step. Eagle3 fuses features from multiple depths (e.g., input embedding, layer `N/2`, and final layer `N`).
- **Output:** Next token logits.

### 2.2 Data Flow
```
Step t:
  Target Model produces: Token `t`, Features `F_t` (at various layers)
  
  Eagle Loop (runs K times on CPU/GPU, very fast):
    Input: Token `t`, Features `F_t`
    Eagle Head -> Token `t+1`, Hidden State `h_{t+1}`
    Input: Token `t+1`, Hidden State `h_{t+1}`
    Eagle Head -> Token `t+2`, Hidden State `h_{t+2}`
    ...
    Result: A tree of K candidates.

Step t+1:
  Target Model Input: Flattened Tree of K tokens.
  Attention Mask: Tree Mask (tokens only attend to their ancestors).
  Target Model Output: Logits for all K positions.
  
  Verification:
    Compare Target(Token `t+i`) vs Eagle(Token `t+i`).
    Accept tokens until divergence.
```

## 3. Implementation Steps for Inference Engine

To implement Eagle3 in a custom engine (like `runq.c` or a Rust-based fuser), follow these steps:

### Phase 1: Model Loading & Tensor Management
1.  **Load Eagle Weights:**
    - The Eagle head is a separate set of weights (usually small, ~0.5B params or less).
    - Format: Standard transformer weights (attention `wq,wk,wv,wo`, FFN `w1,w2,w3`, RMSNorms).
    - *Note:* Eagle3 often requires specific "feature indices" from the target model config (which layers to extract).

2.  **RunState Expansion:**
    - Add buffers to store **Target Model Features**.
    - If Eagle3 uses layers [0, 16, 32], you must save the output of these layers during the Target Model's forward pass.
    - Buffer size: `[feature_dim]` per saved layer.

### Phase 2: The Eagle Draft Loop
1.  **Drafting Function:**
    - Implement a mini-forward pass for the Eagle head.
    - This runs *autoregressively* for `K` steps (e.g., K=4 to 8).
    - **Tree Construction:** Eagle3 uses a dynamic tree (often static top-k expansion in simpler versions). For v1 implementation, a linear chain (top-1) is easiest, but a tree (top-k at each step) yields better speedups.
    - **Output:** A list of `K` candidate tokens and their parent indices (to build the tree structure).

### Phase 3: Target Model Verification (Tree Attention)
1.  **Input Construction:**
    - Construct a batch of input tokens from the draft tree.
    - *Crucial:* The Position IDs must match the depth in the tree, not the batch index.
    
2.  **Attention Masking:**
    - Standard causal mask is `triangular`.
    - **Tree Mask:** A token at index `i` can only attend to token `j` if `j` is an ancestor of `i` in the draft tree.
    - *Implementation:* Create a boolean mask `[K, K]` where `mask[i, j] = 1` if `j` in `ancestors(i)`.

3.  **Forward Pass:**
    - Run the heavy Target Model on these `K` tokens in parallel.
    - Extract logits for all `K` positions.

### Phase 4: Acceptance & KV Cache Management
1.  **Verification Logic:**
    - Iterate through the tree. For each node `i` (predicting token `X`), check if Target Model at parent of `i` actually predicts `X` (greedy) or if `X` is valid under sampling (top-p).
    - Find the longest chain of accepted tokens.

2.  **KV Cache Update:**
    - **Drafting:** During drafting, the Eagle head has its own KV cache.
    - **Target Verification:** The Target Model writes KV entries for *all* candidate tokens into a temporary space or appends speculatively.
    - **Rollback:** If speculation fails at depth `d < K`, you must discard the KV cache entries for the invalid branches and keep only the valid prefix.
    - *Optimization:* PagedAttention (implemented in vLLM) makes this efficient (just free the pages of invalid branches). In a simple C buffer, you might need to `memcpy` or just reset the "length" pointer if using a linear cache.

## 4. Key Challenges & Optimizations

- **Feature Extraction:** You must modify the Target Model's `forward` function to return intermediate activations, not just the final logits.
- **Tree Attention Kernel:** Standard FlashAttention might need a custom mask support or you can use "Block-Diagonal" masking tricks if the tree is simple.
- **Overhead:** The Eagle head must be *fast*. If it takes too long, it defeats the purpose. Run it in fp16/int8.

## 5. Summary of Work
1.  **Loader:** Support loading separate Eagle weights.
2.  **Target Mod:** Expose layer features.
3.  **Draft Mod:** Implement lightweight transformer forward pass.
4.  **Tree Attn:** Implement arbitrary-mask attention in the Target Model.
5.  **Sampler:** Tree verification logic.
