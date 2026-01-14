use cubecl_hip_sys::*;
use std::ffi::{CString, CStr};
use std::ptr;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    unsafe {
        // --- Init ---
        let mut count: std::os::raw::c_int = 0;
        hipGetDeviceCount(&mut count);
        if count == 0 {
            println!("No devices found.");
            return Ok(())
        }
        hipSetDevice(0);
        println!("Using Device 0");

        // --- Kernel Source ---
        // Simple naive MatMul: C = A * B
        // A, B, C are N x N matrices flattened.
        let source_code = r#" 
extern "C" __global__ void matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"#;
        let source = CString::new(source_code).expect("Kernel string");
        let func_name = CString::new("matmul").unwrap();

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

        // --- Prepare Data ---
        let n: i32 = 256; // 256x256 matrix
        let size_elements = (n * n) as usize;
        let size_bytes = size_elements * std::mem::size_of::<f32>();

        // Initialize host matrices (row-major)
        let h_a: Vec<f32> = (0..size_elements).map(|i| (i % n as usize) as f32).collect(); // 0, 1, 2...
        // Identity matrix for B to make verification easy: B[i, j] = 1 if i==j else 0
        let h_b: Vec<f32> = (0..size_elements).map(|i| {
            let r = i / n as usize;
            let c = i % n as usize;
            if r == c { 1.0 } else { 0.0 }
        }).collect();
        let mut h_c: Vec<f32> = vec![0.0; size_elements];

        // Device Alloc
        let mut d_a: *mut libc::c_void = ptr::null_mut();
        let mut d_b: *mut libc::c_void = ptr::null_mut();
        let mut d_c: *mut libc::c_void = ptr::null_mut();

        hipMalloc(&mut d_a, size_bytes);
        hipMalloc(&mut d_b, size_bytes);
        hipMalloc(&mut d_c, size_bytes);

        // Copy H2D
        hipMemcpy(d_a, h_a.as_ptr() as *const _, size_bytes, hipMemcpyKind_hipMemcpyHostToDevice);
        hipMemcpy(d_b, h_b.as_ptr() as *const _, size_bytes, hipMemcpyKind_hipMemcpyHostToDevice);

        // --- Launch ---
        let block_size = 16;
        let grid_size = (n as u32 + block_size - 1) / block_size;

        let mut args: [*mut libc::c_void; 4] = [
            &d_a as *const _ as *mut _,
            &d_b as *const _ as *mut _,
            &d_c as *const _ as *mut _,
            &n as *const _ as *mut _,
        ];

        println!("Launching MatMul ({}x{}) with grid ({},{}), block ({},{})", 
                 n, n, grid_size, grid_size, block_size, block_size);
        
        let start = Instant::now();
        let status = hipModuleLaunchKernel(
            function,
            grid_size, grid_size, 1, // Grid
            block_size, block_size, 1, // Block
            0, ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        );
        hipDeviceSynchronize();
        let duration = start.elapsed();

        if status != HIP_SUCCESS {
            anyhow::bail!("Launch failed: {:?}", status);
        }
        println!("Execution time: {}µs", duration.as_micros());

        // --- Verify ---
        hipMemcpy(h_c.as_mut_ptr() as *mut _, d_c, size_bytes, hipMemcpyKind_hipMemcpyDeviceToHost);

        // Since B is identity, C should equal A
        let mut errors = 0;
        for i in 0..size_elements {
            if (h_c[i] - h_a[i]).abs() > 1e-4 {
                if errors < 5 {
                     println!("Mismatch at {}: expected {}, got {}", i, h_a[i], h_c[i]);
                }
                errors += 1;
            }
        }

        if errors == 0 {
            println!("SUCCESS: MatMul verified (Identiy check).");
        } else {
            println!("FAILURE: {} errors.", errors);
        }

        // Cleanup
        hipFree(d_a);
        hipFree(d_b);
        hipFree(d_c);
        hipModuleUnload(module);
    }
    Ok(())
}
