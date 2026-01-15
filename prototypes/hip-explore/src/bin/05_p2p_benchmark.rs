#![allow(unsafe_op_in_unsafe_fn)]
use cubecl_hip_sys::*;
use std::ptr;

// Helper macro for checking HIP errors
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
        let mut count: std::os::raw::c_int = 0;
        check!(hipGetDeviceCount(&mut count));
        println!("Found {} devices.", count);

        if count < 2 {
            println!("Need at least 2 devices for P2P benchmark.");
            return Ok(());
        }

        // --- Step 1: Enable P2P Access ---
        println!("Checking and enabling P2P access...");
        let mut p2p_matrix = vec![vec![false; count as usize]; count as usize];

        for src in 0..count {
            check!(hipSetDevice(src));
            for dst in 0..count {
                if src == dst {
                    continue;
                }

                let mut can_access: i32 = 0;
                check!(hipDeviceCanAccessPeer(&mut can_access, src, dst));

                if can_access != 0 {
                    let err = hipDeviceEnablePeerAccess(dst, 0);
                    if err == HIP_SUCCESS {
                        p2p_matrix[src as usize][dst as usize] = true;
                    } else {
                        // Some errors are non-fatal (already enabled etc), but good to note
                        println!("Failed to enable P2P from {} to {}: {:?}", src, dst, err);
                    }
                }
            }
        }

        // Print Matrix
        print!("   ");
        for i in 0..count {
            print!(" D{} ", i);
        }
        println!();
        for src in 0..count {
            print!("D{} ", src);
            for dst in 0..count {
                let s = if src == dst {
                    " - "
                } else if p2p_matrix[src as usize][dst as usize] {
                    " Y "
                } else {
                    " N "
                };
                print!("{}", s);
            }
            println!();
        }
        println!();

        // --- Step 2: Benchmarks ---
        let small_size = 4; // 4 bytes for latency
        let large_size = 256 * 1024 * 1024; // 256 MB for bandwidth

        for src in 0..count {
            for dst in 0..count {
                if src == dst {
                    continue;
                }
                if !p2p_matrix[src as usize][dst as usize] {
                    continue; // Skip if no P2P (likely very slow or not supported)
                }

                println!("Benchmarking D{} -> D{}...", src, dst);
                run_benchmark(src, dst, small_size, large_size)?;
            }
        }

        // --- Step 3: Bidirectional (Bonus) ---
        // Just checking pairs (0,1), (2,3) etc if available
        println!("\nBidirectional Benchmarks:");
        for i in 0..count {
            for j in (i + 1)..count {
                if p2p_matrix[i as usize][j as usize] && p2p_matrix[j as usize][i as usize] {
                    run_bidirectional(i, j, large_size)?;
                }
            }
        }
    }
    Ok(())
}

unsafe fn run_benchmark(
    src_dev: i32,
    dst_dev: i32,
    small_size: usize,
    large_size: usize,
) -> anyhow::Result<()> {
    // Alloc
    let mut d_src: *mut libc::c_void = ptr::null_mut();
    let mut d_dst: *mut libc::c_void = ptr::null_mut();

    check!(hipSetDevice(src_dev));
    check!(hipMalloc(&mut d_src, large_size));
    check!(hipMemset(d_src, 1, large_size)); // Initialize

    check!(hipSetDevice(dst_dev));
    check!(hipMalloc(&mut d_dst, large_size));

    check!(hipSetDevice(src_dev)); // Master context
    let mut stream: hipStream_t = ptr::null_mut();
    check!(hipStreamCreate(&mut stream));

    let mut start_evt: hipEvent_t = ptr::null_mut();
    let mut stop_evt: hipEvent_t = ptr::null_mut();
    check!(hipEventCreate(&mut start_evt));
    check!(hipEventCreate(&mut stop_evt));

    // --- Latency ---
    let iterations = 100;
    check!(hipEventRecord(start_evt, stream));
    for _ in 0..iterations {
        check!(hipMemcpyPeerAsync(
            d_dst, dst_dev, d_src, src_dev, small_size, stream
        ));
    }
    check!(hipEventRecord(stop_evt, stream));
    check!(hipEventSynchronize(stop_evt));

    let mut ms: f32 = 0.0;
    check!(hipEventElapsedTime(&mut ms, start_evt, stop_evt));
    let latency_us = (ms * 1000.0) / (iterations as f32);

    // --- Bandwidth ---
    // Warmup
    check!(hipMemcpyPeerAsync(
        d_dst, dst_dev, d_src, src_dev, large_size, stream
    ));
    check!(hipStreamSynchronize(stream));

    let bw_iterations = 10;
    check!(hipEventRecord(start_evt, stream));
    for _ in 0..bw_iterations {
        check!(hipMemcpyPeerAsync(
            d_dst, dst_dev, d_src, src_dev, large_size, stream
        ));
    }
    check!(hipEventRecord(stop_evt, stream));
    check!(hipEventSynchronize(stop_evt));

    let mut ms_bw: f32 = 0.0;
    check!(hipEventElapsedTime(&mut ms_bw, start_evt, stop_evt));

    let total_bytes = (large_size as f64) * (bw_iterations as f64);
    let seconds = (ms_bw as f64) / 1000.0;
    let gb_s = (total_bytes / seconds) / (1024.0 * 1024.0 * 1024.0);

    println!("  Latency: {:.2} us", latency_us);
    println!("  Bandwidth: {:.2} GB/s", gb_s);

    // Cleanup
    check!(hipEventDestroy(start_evt));
    check!(hipEventDestroy(stop_evt));
    check!(hipStreamDestroy(stream));
    // d_src was allocated on src_dev.
    check!(hipSetDevice(src_dev));
    check!(hipFree(d_src));
    check!(hipSetDevice(dst_dev));
    check!(hipFree(d_dst));

    Ok(())
}

unsafe fn run_bidirectional(dev_a: i32, dev_b: i32, size: usize) -> anyhow::Result<()> {
    println!("Bidirectional D{} <-> D{}...", dev_a, dev_b);

    let mut d_a: *mut libc::c_void = ptr::null_mut(); // On A
    let mut d_b: *mut libc::c_void = ptr::null_mut(); // On B
    // Buffers for return trip
    let mut d_a_recv: *mut libc::c_void = ptr::null_mut(); // On A
    let mut d_b_recv: *mut libc::c_void = ptr::null_mut(); // On B

    check!(hipSetDevice(dev_a));
    check!(hipMalloc(&mut d_a, size));
    check!(hipMalloc(&mut d_a_recv, size));

    check!(hipSetDevice(dev_b));
    check!(hipMalloc(&mut d_b, size));
    check!(hipMalloc(&mut d_b_recv, size));

    // Two streams, one per device/direction ideally to ensure concurrency
    check!(hipSetDevice(dev_a));
    let mut stream_a: hipStream_t = ptr::null_mut();
    check!(hipStreamCreate(&mut stream_a));
    let mut start_a: hipEvent_t = ptr::null_mut();
    let mut stop_a: hipEvent_t = ptr::null_mut();
    check!(hipEventCreate(&mut start_a));
    check!(hipEventCreate(&mut stop_a));

    check!(hipSetDevice(dev_b));
    let mut stream_b: hipStream_t = ptr::null_mut();
    check!(hipStreamCreate(&mut stream_b));
    let mut start_b: hipEvent_t = ptr::null_mut();
    let mut stop_b: hipEvent_t = ptr::null_mut();
    check!(hipEventCreate(&mut start_b));
    check!(hipEventCreate(&mut stop_b));

    let iterations = 5;

    // Start timing
    // We synchronize CPU to start approx same time, but use events for duration.
    // For bidirectional, we want Total Bytes / Max(TimeA, TimeB).

    // A -> B (Stream A)
    check!(hipSetDevice(dev_a));
    check!(hipEventRecord(start_a, stream_a));
    for _ in 0..iterations {
        check!(hipMemcpyPeerAsync(
            d_b_recv, dev_b, d_a, dev_a, size, stream_a
        ));
    }
    check!(hipEventRecord(stop_a, stream_a));

    // B -> A (Stream B)
    check!(hipSetDevice(dev_b));
    check!(hipEventRecord(start_b, stream_b));
    for _ in 0..iterations {
        check!(hipMemcpyPeerAsync(
            d_a_recv, dev_a, d_b, dev_b, size, stream_b
        ));
    }
    check!(hipEventRecord(stop_b, stream_b));

    // Wait
    check!(hipSetDevice(dev_a));
    check!(hipEventSynchronize(stop_a));
    check!(hipSetDevice(dev_b));
    check!(hipEventSynchronize(stop_b));

    let mut ms_a: f32 = 0.0;
    let mut ms_b: f32 = 0.0;
    check!(hipEventElapsedTime(&mut ms_a, start_a, stop_a));
    check!(hipEventElapsedTime(&mut ms_b, start_b, stop_b));

    let max_ms = if ms_a > ms_b { ms_a } else { ms_b };
    let total_bytes = (size as f64 * iterations as f64) * 2.0; // Both directions
    let gb_s = (total_bytes / (max_ms as f64 / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    println!("  Time A->B: {:.2} ms", ms_a);
    println!("  Time B->A: {:.2} ms", ms_b);
    println!("  Aggregate Bandwidth: {:.2} GB/s", gb_s);

    // Clean
    check!(hipStreamDestroy(stream_a));
    check!(hipStreamDestroy(stream_b));
    // ... free memory ...
    Ok(())
}
