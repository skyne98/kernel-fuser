# vLLM Paged KV Cache and PagedAttention

This document details the architecture and implementation of vLLM's Paged KV Cache (PagedAttention), a technique that significantly optimizes memory usage and throughput for Large Language Model (LLM) inference.

## 1. Core Concept: The Memory Fragmentation Problem

Traditional LLM serving systems require contiguous memory allocation for the Key-Value (KV) cache of each request. Because the length of a generated sequence is unknown beforehand, systems typically pre-allocate memory based on a maximum context length (e.g., 2048 or 4096 tokens).

This leads to three types of memory waste:
1.  **Reserved (Internal Fragmentation):** Memory reserved for the maximum sequence length but never used if the sequence stops early.
2.  **Internal Fragmentation:** Memory allocated for a request but not yet populated as generation proceeds token-by-token.
3.  **External Fragmentation:** Memory gaps between allocations that are too small to serve new requests.

vLLM solves this by adopting **PagedAttention**, an algorithm inspired by virtual memory paging in operating systems.

## 2. PagedAttention Architecture

PagedAttention decouples **logical** KV blocks (what the model sees) from **physical** KV blocks (where data actually lives in VRAM).

### 2.1. Logical vs. Physical Blocks
- **Logical Blocks:** The KV cache for a sequence is divided into logical blocks of a fixed size (e.g., 16 or 32 tokens). From the perspective of the attention mechanism, these blocks appear contiguous.
- **Physical Blocks:** These are the actual storage units in GPU memory. They are allocated on-demand and can be stored non-contiguously.

### 2.2. The Block Table
Similar to a page table in an OS, vLLM maintains a **Block Table** for each sequence.
- **Mapping:** It maps logical block indices to physical block indices.
- **Dynamic Growth:** As a sequence generates new tokens, new physical blocks are allocated from a global free pool and added to the block table.
- **No Pre-allocation:** Memory is only consumed for the tokens actually generated, reducing waste from nearly 60-80% to under 4%.

## 3. Implementation Details: Control Plane (Python)

The control plane manages the state of memory blocks and schedules requests. This logic resides primarily in the `vllm.core` module.

### 3.1. BlockSpaceManager (`vllm.core.block_manager.py`)
This is the central memory allocator. It manages the global pool of physical blocks (both on GPU and CPU for swapping).
- **Responsibilities:**
    - Allocating physical blocks for new requests.
    - Freeing blocks when requests finish.
    - Tracking reference counts for blocks (crucial for shared prefixes).
    - Managing "swapping" of blocks to CPU RAM when GPU memory is full (preemption).
- **Lookahead Slots:** It supports speculative decoding by managing "lookahead" allocations for proposal tokens.

### 3.2. BlockTable (`vllm.core.block.block_table.py`)
This class represents the memory state of a single sequence.
- **Mapping:** Stores the list of `PhysicalTokenBlock` objects assigned to the sequence.
- **Methods:**
    - `allocate()`: Reserves new blocks.
    - `append_token_ids()`: Adds new tokens to the current block or triggers allocation of a new one.
- **Forking:** When doing beam search or parallel sampling, a sequence can "fork". The new child sequence initially shares the same physical blocks as the parent (Copy-on-Write).

## 4. Implementation Details: Data Plane (CUDA Kernels)

The actual computation happens in custom CUDA kernels, specifically optimized to understand this block-indirection layer.

### 4.1. The PagedAttention Kernel (`csrc/attention/attention_kernels.cu`)
Standard attention kernels (like FlashAttention) assume contiguous memory. vLLM's custom kernel handles the non-contiguous fetch.

- **Inputs:**
    - `q`: Query vectors.
    - `k_cache`, `v_cache`: Pointers to the global physical block storage.
    - `block_tables`: A tensor passing the block mapping to the GPU.
- **Execution Flow:**
    1.  **Block Retrieval:** For a given sequence and query token, the kernel looks up the `block_tables` to find the physical addresses of the relevant KV blocks.
    2.  **Warp-Level Parallelism:**
        - A **Warp** (32 threads) usually handles the attention for a single query token against a single KV block (or multiple, depending on optimization).
        - Threads cooperate to load keys/values from the scattered physical blocks into shared memory or registers.
    3.  **Compute:** Dot products (Q · K) and aggregations (Softmax) are performed.
    4.  **Output:** The final context vector is computed and written to global memory.

## 5. Advanced Optimizations

### 5.1. Memory Sharing & Copy-on-Write
vLLM efficiently handles complex decoding scenarios (like parallel sampling or beam search) and shared system prompts.
- **Shared Blocks:** If multiple sequences start with the same prompt, their block tables point to the same physical blocks. The reference count for those physical blocks > 1.
- **Copy-on-Write (CoW):** If a sequence needs to write to a shared block (e.g., appending a new token to a partially full shared block), the `BlockSpaceManager` creates a copy of that block for the specific sequence, ensuring isolation while maximizing sharing.

### 5.2. Swapping
When the GPU runs out of blocks, the scheduler can "preempt" a lower-priority sequence.
- Its blocks are swapped out to CPU memory (pinned memory).
- The Block Table is updated to reflect that these blocks are now on the CPU.
- Later, they are swapped back in to resume generation.

## 6. Summary of Benefits

- **Near-Zero Waste:** Eliminates internal fragmentation from pre-allocation.
- **Higher Batch Sizes:** Saved memory allows fitting more concurrent sequences, directly increasing throughput.
- **Flexible Decoding:** Efficiently supports complex sampling methods (beam search) via memory sharing.
