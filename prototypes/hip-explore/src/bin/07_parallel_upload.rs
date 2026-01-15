#![allow(unsafe_op_in_unsafe_fn)]
use cubecl_hip_sys::*;
use std::ptr;
use std::thread;
use std::time::Instant;

macro_rules! check {
    ($e:expr) => {{
        let res = $e;
        if res != HIP_SUCCESS {
            anyhow::bail!("HIP Error at line {}: {:?}", line!(), res);
        }
    }};
}

fn main() -> anyhow::Result<()> {
    unsafe {
        let mut n_gpu: i32 = 0;
        check!(hipGetDeviceCount(&mut n_gpu));
        let n_cpu = std::thread::available_parallelism()?.get();

        println!("System Info:");
        println!("  GPUs: {}", n_gpu);
        println!("  CPUs: {}", n_cpu);

        if n_gpu == 0 {
            anyhow::bail!("No GPUs found.");
        }

        let threads_per_gpu = (n_cpu as f32 / n_gpu as f32).max(1.0).floor() as usize;
        let transfer_size_per_gpu = 1024 * 1024 * 1024; // 1 GB
        let total_size = transfer_size_per_gpu * n_gpu as usize;

        // --- Step 1: Allocate Pinned Host Memory ---
        let mut h_ptr: *mut libc::c_void = ptr::null_mut();
        check!(hipHostMalloc(&mut h_ptr, total_size, hipHostMallocDefault));

        let host_slice = std::slice::from_raw_parts_mut(h_ptr as *mut u8, total_size);
        host_slice.fill(0xAB);

        // --- Step 2: Allocate Device Memory ---
        let mut d_ptrs = Vec::new();
        for i in 0..n_gpu {
            check!(hipSetDevice(i));
            let mut d_ptr: *mut libc::c_void = ptr::null_mut();
            check!(hipMalloc(&mut d_ptr, transfer_size_per_gpu));
            d_ptrs.push(d_ptr as usize);
        }

        // --- PHASE 1: Serial Upload ---
        println!("\nPHASE 1: Serial Upload (1 thread, 1 GPU at a time)");
        let start_serial = Instant::now();
        for i in 0..n_gpu {
            check!(hipSetDevice(i));
            check!(hipMemcpy(
                d_ptrs[i as usize] as *mut _,
                h_ptr.add(i as usize * transfer_size_per_gpu),
                transfer_size_per_gpu,
                hipMemcpyKind_hipMemcpyHostToDevice
            ));
        }
        let duration_serial = start_serial.elapsed();
        let bw_serial =
            (total_size as f64 / 1024.0 / 1024.0 / 1024.0) / duration_serial.as_secs_f64();
        println!("  Bandwidth: {:.2} GB/s", bw_serial);

        // --- PHASE 2: Parallel Upload (Distributed) ---
        println!(
            "\nPHASE 2: Distributed Parallel Upload ({} threads per GPU, all GPUs)",
            threads_per_gpu
        );
        let bw_parallel = run_upload_benchmark(
            n_gpu,
            threads_per_gpu,
            h_ptr,
            &d_ptrs,
            transfer_size_per_gpu,
            true,
        )?;
        println!("  Aggregate Bandwidth: {:.2} GB/s", bw_parallel);

        // --- PHASE 3: Single GPU, All Threads ---
        println!(
            "\nPHASE 3: Single-GPU Saturation (All {} threads -> GPU 0)",
            n_cpu
        );
        let bw_single =
            run_upload_benchmark(1, n_cpu, h_ptr, &d_ptrs[0..1], transfer_size_per_gpu, false)?;
        println!("  Single GPU Bandwidth: {:.2} GB/s", bw_single);

        // --- Cleanup ---
        for i in 0..n_gpu {
            check!(hipSetDevice(i));
            check!(hipFree(d_ptrs[i as usize] as *mut _));
        }
        check!(hipHostFree(h_ptr));
    }
    Ok(())
}

unsafe fn run_upload_benchmark(
    n_gpu_to_use: i32,
    threads_per_gpu: usize,
    h_base: *mut libc::c_void,
    d_bases: &[usize],
    size_per_gpu: usize,
    offset_host: bool,
) -> anyhow::Result<f64> {
    let start = Instant::now();
    let mut handles = Vec::new();
    let chunk_size = size_per_gpu / threads_per_gpu;

    for g in 0..n_gpu_to_use {
        for t in 0..threads_per_gpu {
            let gpu_id = g;
            let h_offset = if offset_host {
                (g as usize * size_per_gpu) + (t * chunk_size)
            } else {
                t * chunk_size
            };
            let src = h_base.add(h_offset) as usize;
            let dst = d_bases[g as usize] + (t * chunk_size);
            let size = if t == threads_per_gpu - 1 {
                size_per_gpu - (t * chunk_size)
            } else {
                chunk_size
            };

            handles.push(thread::spawn(move || -> anyhow::Result<()> {
                unsafe {
                    check!(hipSetDevice(gpu_id));
                    let mut stream: hipStream_t = ptr::null_mut();
                    check!(hipStreamCreate(&mut stream));
                    check!(hipMemcpyAsync(
                        dst as *mut _,
                        src as *const _,
                        size,
                        hipMemcpyKind_hipMemcpyHostToDevice,
                        stream
                    ));
                    check!(hipStreamSynchronize(stream));
                    check!(hipStreamDestroy(stream));
                }
                Ok(())
            }));
        }
    }

    for handle in handles {
        handle.join().expect("Thread panicked")?;
    }

    let duration = start.elapsed();
    let total_bytes = size_per_gpu * n_gpu_to_use as usize;
    Ok((total_bytes as f64 / 1024.0 / 1024.0 / 1024.0) / duration.as_secs_f64())
}
