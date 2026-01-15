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

### 1. Simple Vector Add (Raw)
```bash
cargo run --bin 01_simple_add
```
Uses raw `cubecl-hip-sys` bindings and manual resource management.

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

### 7. Parallel Upload Benchmark
```bash
cargo run --bin 07_parallel_upload
```
Tests serial vs distributed vs single-GPU multithreaded upload speeds.

### 8. Simple Vector Add (Safe Wrapper)
```bash
cargo run --bin 08_simple_add_safe
```
Identical to binary 01, but uses the safe Rust wrapper in `src/lib.rs` for automatic resource cleanup and error handling.

## Features
- **Safe Wrapper:** `src/lib.rs` provides RAII types (`DeviceBuffer`, `Stream`, `Module`, etc.) for HIP.
- **Dynamic Linking:** Handled via `build.rs`.
- **Runtime Compilation:** (HIPRTC) of C++ kernels.
