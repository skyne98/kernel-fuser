use cubecl_hip_sys::*;
use std::ffi::{CStr, CString};
use std::ptr;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    unsafe {
        // --- Init ---
        let mut count: std::os::raw::c_int = 0;
        hipGetDeviceCount(&mut count);
        if count == 0 {
            println!("No devices found.");
            return Ok(());
        }
        hipSetDevice(0);
        println!("Using Device 0");

        // --- Kernel Source ---
        // Simple kernel: out[i] = in[i] * 2.0
        let source_code = r#"extern "C" __global__ void scale_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}
"#;
        let source = CString::new(source_code).expect("Kernel string");
        let func_name = CString::new("scale_kernel").unwrap();

        // --- Compile (HIPRTC) ---
        let mut program: hiprtcProgram = ptr::null_mut();
        hiprtcCreateProgram(
            &mut program,
            source.as_ptr(),
            ptr::null(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
        );

        let status = hiprtcCompileProgram(program, 0, ptr::null_mut());
        if status != hiprtcResult_HIPRTC_SUCCESS {
            let mut log_size: usize = 0;
            hiprtcGetProgramLogSize(program, &mut log_size);
            let mut log_buffer = vec![0i8; log_size];
            hiprtcGetProgramLog(program, log_buffer.as_mut_ptr());
            let log = CStr::from_ptr(log_buffer.as_ptr());
            anyhow::bail!("Compilation failed:\n{}", log.to_string_lossy());
        }

        let mut code_size: usize = 0;
        hiprtcGetCodeSize(program, &mut code_size);
        let mut code: Vec<u8> = vec![0; code_size];
        hiprtcGetCode(program, code.as_mut_ptr() as *mut _);
        hiprtcDestroyProgram(&mut program);

        // --- Load Module ---
        let mut module: hipModule_t = ptr::null_mut();
        let mut function: hipFunction_t = ptr::null_mut();
        hipModuleLoadData(&mut module, code.as_ptr() as *const _);
        hipModuleGetFunction(&mut function, module, func_name.as_ptr());

        // --- Managed Memory Allocation ---
        let n: usize = 1024 * 1024; // 1M floats = 4MB
        let size_bytes = n * std::mem::size_of::<f32>();

        let mut d_data: *mut libc::c_void = ptr::null_mut();

        // Allocate Managed Memory
        // hipMallocManaged(&ptr, size, flags)
        // flags: hipMemAttachGlobal = 1
        println!("Allocating {} bytes of Managed Memory...", size_bytes);
        let status = hipMallocManaged(&mut d_data, size_bytes, hipMemAttachGlobal);
        if status != HIP_SUCCESS {
            anyhow::bail!("hipMallocManaged failed: {:?}", status);
        }

        // --- Host Access (Initialization) ---
        // We interpret the pointer as a slice and write to it directly from the CPU.
        // This 'pages in' the memory to the Host (or just initializes it in system RAM).
        let host_slice = std::slice::from_raw_parts_mut(d_data as *mut f32, n);
        println!("Initializing data on HOST...");
        for i in 0..n {
            host_slice[i] = i as f32;
        }

        // --- Device Access (Kernel Execution) ---
        // Launch kernel using the SAME pointer.
        // The GPU will access these pages. If XNACK is enabled and supported,
        // it triggers page faults on the GPU, migrating pages from Host to Device VRAM.

        let block_size = 256;
        let grid_size = (n as u32 + block_size - 1) / block_size;
        let n_c = n as i32;

        let mut args: [*mut libc::c_void; 2] =
            [&d_data as *const _ as *mut _, &n_c as *const _ as *mut _];

        println!("Launching kernel on DEVICE (accessing managed memory)...");
        let start = Instant::now();
        let status = hipModuleLaunchKernel(
            function,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        );
        if status != HIP_SUCCESS {
            anyhow::bail!("Launch failed: {:?}", status);
        }

        // Synchronize to ensure kernel is done
        hipDeviceSynchronize();
        let duration = start.elapsed();
        println!("Kernel execution time: {}µs", duration.as_micros());

        // --- Host Access (Verification) ---
        // Read back the data. Pages might migrate back to Host (or be accessed over PCIe).
        println!("Verifying data on HOST...");
        let mut errors = 0;
        for i in 0..n {
            let expected = (i as f32) * 2.0;
            let got = host_slice[i];
            if (got - expected).abs() > 1e-5 {
                if errors < 5 {
                    println!("Mismatch at {}: expected {}, got {}", i, expected, got);
                }
                errors += 1;
            }
        }

        if errors == 0 {
            println!("SUCCESS: Managed Memory consistency verified.");
        } else {
            println!("FAILURE: {} errors.", errors);
        }

        // Cleanup
        hipFree(d_data);
        hipModuleUnload(module);
    }
    Ok(())
}
