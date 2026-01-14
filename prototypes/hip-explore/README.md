# HIP Explore Prototype

This prototype explores usage of `cubecl-hip-sys` to compile and run kernels on AMD GPUs via HIP.

## Prerequisites

- AMD ROCm installed (tested with HIP 5.7)
- `hipcc` available in PATH.

## Setup

This project uses `cubecl-hip-sys` from crates.io.
To handle the version mismatch (System HIP 5.7 vs Crate expectation), we enable the `hip_51831` feature via `RUSTFLAGS` in `.cargo/config.toml` at the workspace root.

**Note for IDEs:**
Most IDEs (VSCode with rust-analyzer, IntelliJ Rust) should automatically pick up the configuration from `.cargo/config.toml`. If you still see "symbol not found" errors, try reloading the workspace or restarting the language server.

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

## Features
- Dynamic library linking via `build.rs`.
- Runtime compilation (HIPRTC) of C++ kernels.
