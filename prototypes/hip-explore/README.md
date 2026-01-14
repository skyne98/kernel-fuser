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
- Allocates memory accessible to both Host and Device.
- Initializes on Host (no explicit copy).
- Kernel modifies data on Device (pages migrate/fault on demand).
- Host verifies results (pages accessible again).

### 4. MMAP File Access (NVMe -> CPU -> GPU)
```bash
HSA_XNACK=1 cargo run --bin 04_mmap_lazy
```
Demonstrates "Zero-Copy" access to a file on disk (NVMe).
- Creates a large file (256MB).
- Maps it into Host memory (`mmap`).
- **Registers** the pointer with HIP (`hipHostRegister`).
  - *Note:* On systems without full HMM support (like the MI50 tested), this registration pins the memory, forcing a load from disk to RAM. True lazy loading requires `PageableMemoryAccess` support.
- GPU reads directly from Host RAM (which mirrors the file).
- Reads specific bytes to verify connectivity.

## Features
- Dynamic library linking via `build.rs`.
- Runtime compilation (HIPRTC) of C++ kernels.
- Multiple example kernels covering standard, managed, and mmap usage.