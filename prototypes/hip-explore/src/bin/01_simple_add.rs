use cubecl_hip_sys::*;
use std::ffi::{CStr, CString};
use std::ptr;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    unsafe {
        // --- Step 0: Discovery ---
        let mut count: std::os::raw::c_int = 0;
        let result = hipGetDeviceCount(&mut count);
        if result != HIP_SUCCESS {
            anyhow::bail!("Failed to get device count: {:?}", result);
        }
        println!("Found {} HIP devices.", count);

        if count == 0 {
            println!("No devices found, exiting.");
            return Ok(());
        }

        // Use device 0
        let device_id = 0;
        println!("Using device {}", device_id);
        let status = hipSetDevice(device_id);
        if status != HIP_SUCCESS {
            anyhow::bail!("Failed to set device: {:?}", status);
        }

        // Check memory info
        let mut free: usize = 0;
        let mut total: usize = 0;
        let status = hipMemGetInfo(&mut free, &mut total);
        if status != HIP_SUCCESS {
            anyhow::bail!("Failed to get memory info: {:?}", status);
        }
        println!(
            "Device Memory: Free: {} bytes | Total: {} bytes",
            free, total
        );

        // --- Step 1: Kernel Source ---
        // Kernel that computes out = x * a + b
        let source_code = r#"
extern "C" __global__ void kernel(float a, float *x, float *b, float *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = x[tid] * a + b[tid];
  }
}
"#;
        let source = CString::new(source_code).expect("Should construct kernel string");
        let func_name = CString::new("kernel").unwrap();

        // --- Step 2: Create Program (HIPRTC) ---
        println!("Creating HIPRTC program...");
        let mut program: hiprtcProgram = ptr::null_mut();
        let status = hiprtcCreateProgram(
            &mut program,    // Program
            source.as_ptr(), // kernel string
            ptr::null(),     // Name of the file
            0,               // Number of headers
            ptr::null_mut(), // Header sources
            ptr::null_mut(), // Name of header files
        );
        if status != hiprtcResult_HIPRTC_SUCCESS {
            anyhow::bail!("hiprtcCreateProgram failed: {:?}", status);
        }

        // --- Step 3: Compile Program ---
        println!("Compiling kernel...");
        let status = hiprtcCompileProgram(program, 0, ptr::null_mut());

        if status != hiprtcResult_HIPRTC_SUCCESS {
            // Get log
            let mut log_size: usize = 0;
            hiprtcGetProgramLogSize(program, &mut log_size);
            let mut log_buffer = vec![0i8; log_size];
            hiprtcGetProgramLog(program, log_buffer.as_mut_ptr());
            let log = CStr::from_ptr(log_buffer.as_ptr());
            eprintln!("Compilation failed. Log:\n{}", log.to_string_lossy());
            anyhow::bail!("hiprtcCompileProgram failed: {:?}", status);
        }

        // --- Step 4: Get Compiled Code ---
        let mut code_size: usize = 0;
        hiprtcGetCodeSize(program, &mut code_size);
        let mut code: Vec<u8> = vec![0; code_size];
        hiprtcGetCode(program, code.as_mut_ptr() as *mut _);

        hiprtcDestroyProgram(&mut program);
        println!("Kernel compiled. Code size: {} bytes", code_size);

        // --- Step 5: Load Module ---
        let mut module: hipModule_t = ptr::null_mut();
        let mut function: hipFunction_t = ptr::null_mut();

        let status = hipModuleLoadData(&mut module, code.as_ptr() as *const libc::c_void);
        if status != HIP_SUCCESS {
            anyhow::bail!("hipModuleLoadData failed: {:?}", status);
        }

        let status = hipModuleGetFunction(&mut function, module, func_name.as_ptr());
        if status != HIP_SUCCESS {
            anyhow::bail!("hipModuleGetFunction failed: {:?}", status);
        }

        // --- Step 6: Prepare Data ---
        let n: i32 = 1024;
        let a: f32 = 2.0;
        let x_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
        let mut out_host: Vec<f32> = vec![0.0; n as usize];

        // Allocate Device Memory
        let mut d_x: *mut libc::c_void = ptr::null_mut();
        let mut d_b: *mut libc::c_void = ptr::null_mut();
        let mut d_out: *mut libc::c_void = ptr::null_mut();

        let size_bytes = (n as usize) * std::mem::size_of::<f32>();

        hipMalloc(&mut d_x, size_bytes);
        hipMalloc(&mut d_b, size_bytes);
        hipMalloc(&mut d_out, size_bytes);

        // Copy Host to Device
        hipMemcpy(
            d_x,
            x_host.as_ptr() as *const _,
            size_bytes,
            hipMemcpyKind_hipMemcpyHostToDevice,
        );
        hipMemcpy(
            d_b,
            b_host.as_ptr() as *const _,
            size_bytes,
            hipMemcpyKind_hipMemcpyHostToDevice,
        );

        // --- Step 7: Launch Kernel ---
        println!("Launching kernel...");
        let start_time = Instant::now();

        let mut args: [*mut libc::c_void; 5] = [
            &a as *const _ as *mut libc::c_void,
            &d_x as *const _ as *mut libc::c_void,
            &d_b as *const _ as *mut libc::c_void,
            &d_out as *const _ as *mut libc::c_void,
            &n as *const _ as *mut libc::c_void,
        ];

        let block_dim_x = 64;
        let grid_dim_x = (n as u32 + block_dim_x - 1) / block_dim_x;

        let status = hipModuleLaunchKernel(
            function,
            grid_dim_x,
            1,
            1,
            block_dim_x,
            1,
            1,
            0,
            ptr::null_mut(), // stream 0
            args.as_mut_ptr(),
            ptr::null_mut(),
        );
        if status != HIP_SUCCESS {
            anyhow::bail!("Kernel launch failed: {:?}", status);
        }

        hipDeviceSynchronize();
        let duration = start_time.elapsed();
        println!("Kernel finished in {}µs", duration.as_micros());

        // --- Step 8: Retrieve Results ---
        hipMemcpy(
            out_host.as_mut_ptr() as *mut _,
            d_out,
            size_bytes,
            hipMemcpyKind_hipMemcpyDeviceToHost,
        );

        // Verify
        let mut errors = 0;
        for i in 0..n as usize {
            let expected = x_host[i] * a + b_host[i];
            let got = out_host[i];
            if (got - expected).abs() > 1e-5 {
                if errors < 5 {
                    println!("Mismatch at {}: expected {}, got {}", i, expected, got);
                }
                errors += 1;
            }
        }

        if errors == 0 {
            println!("SUCCESS: All results match.");
        } else {
            println!("FAILURE: {} errors found.", errors);
        }

        // Cleanup
        hipFree(d_x);
        hipFree(d_b);
        hipFree(d_out);
        hipModuleUnload(module);
    }

    Ok(())
}
