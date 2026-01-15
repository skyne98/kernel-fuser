# HIP Explore Prototype

This prototype explores usage of `cubecl-hip-sys` to compile and run kernels on AMD GPUs via HIP.

## Prerequisites

- AMD ROCm installed (tested with HIP 5.7)
- `hipcc` available in PATH.
- **For Managed Memory (03) & MMAP (04):** Support for Unified Memory / Page Migration. On MI50/Vega20, this typically requires `HSA_XNACK=1`.

## Setup

This project uses `cubecl-hip-sys` from crates.io.
To handle the version mismatch (System HIP 5.7 vs Crate expectation), we enable the `hip_51831` feature via `RUSTFLAGS` in `.cargo/config.toml` at the workspace root.

## Running

Run specific binaries:

### 1. Simple Vector Add
```bash
cargo run --bin 01_simple_add
```

### 2. Matrix Multiplication
```bash
cargo run --bin 02_matrix_mul
```

### 3. Lazy Managed Memory
```bash
HSA_XNACK=1 cargo run --bin 03_lazy_managed
```

### 4. MMAP File Access
```bash
HSA_XNACK=1 cargo run --bin 04_mmap_lazy
```

### 5. P2P Benchmark
```bash
cargo run --bin 05_p2p_benchmark
```

### 6. Memory & Cache Info
```bash
cargo run --bin 06_memory_info
```
Queries detailed HBM memory capacity, bus widths, L2 cache sizes, and shared memory characteristics for each GPU.

## Features
- Dynamic library linking via `build.rs`.
- Runtime compilation (HIPRTC) of C++ kernels.
- Multiple example kernels covering standard, managed, mmap, P2P, and introspection.