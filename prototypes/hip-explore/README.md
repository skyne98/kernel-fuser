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
Performs a simple `y = ax + b` calculation.

### 2. Matrix Multiplication
```bash
cargo run --bin 02_matrix_mul
```
Performs a naive 256x256 Matrix Multiplication `C = A * B` where B is an Identity matrix, verifying that `C == A`.

### 3. Lazy Managed Memory (True Lazy)
```bash
HSA_XNACK=1 cargo run --bin 03_lazy_managed
```
Demonstrates Unified Memory (`hipMallocManaged`).

### 4. MMAP File Access (NVMe -> CPU -> GPU)
```bash
HSA_XNACK=1 cargo run --bin 04_mmap_lazy
```
Demonstrates "Zero-Copy" access to a file on disk (NVMe).

### 5. P2P Benchmark (Multi-GPU)
```bash
cargo run --bin 05_p2p_benchmark
```
Benchmarks Peer-to-Peer communication between all available GPUs.
- **Latency:** Ping-pong small messages.
- **Bandwidth:** Uni-directional large transfers.
- **Bidirectional:** Concurrent transfers between pairs.
Useful for verifying XGMI/Infinity Fabric links.

## Features
- Dynamic library linking via `build.rs`.
- Runtime compilation (HIPRTC) of C++ kernels.
- Multiple example kernels covering standard, managed, and mmap usage.
