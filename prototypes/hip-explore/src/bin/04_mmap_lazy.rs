use cubecl_hip_sys::*;
use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::io::AsRawFd;
use std::ptr;

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

        // Check for Pageable Memory Access (HMM) support
        let mut pageable_access: i32 = 0;
        hipDeviceGetAttribute(
            &mut pageable_access,
            hipDeviceAttribute_t_hipDeviceAttributePageableMemoryAccess,
            0,
        );
        println!(
            "Pageable Memory Access (HMM): {}",
            if pageable_access != 0 { "YES" } else { "NO" }
        );

        if pageable_access == 0 {
            println!(
                "WARNING: Device 0 does not report pageable memory access support. This demo might crash or fail."
            );
        }

        // --- Prepare File ---
        let filename = "mmap_data.bin";
        let n: usize = 64 * 1024 * 1024; // 64 million floats = 256 MB
        let size_bytes = n * std::mem::size_of::<f32>();

        // create file with some pattern
        println!(
            "Creating/Filling file '{}' ({} MB)...",
            filename,
            size_bytes / (1024 * 1024)
        );
        {
            let mut file = File::create(filename)?;
            // Write chunks to avoid huge memory usage during creation
            let chunk_elem = 1024 * 1024; // 1M floats
            let chunk_bytes = chunk_elem * 4;
            let mut buffer: Vec<u8> = vec![0; chunk_bytes];

            // Fill buffer with simple pattern bytes
            for i in 0..chunk_bytes {
                buffer[i] = (i % 255) as u8;
            }

            let chunks = size_bytes / chunk_bytes;
            for _ in 0..chunks {
                file.write_all(&buffer)?;
            }
        }

        // Open for mmap
        let file = OpenOptions::new().read(true).write(true).open(filename)?;
        let fd = file.as_raw_fd();

        // --- MMAP ---
        println!("Mmapping file...");
        let ptr = libc::mmap(
            ptr::null_mut(),
            size_bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        );

        if ptr == libc::MAP_FAILED {
            anyhow::bail!("mmap failed");
        }

        let f32_ptr = ptr as *mut f32;

        // Verify/Set specific values to check later
        *f32_ptr.add(0) = 1.0;
        *f32_ptr.add(n / 2) = 2.0;
        *f32_ptr.add(n - 1) = 3.0;

        // Register the memory to allow GPU access
        // Note: This likely pins the memory, reading it from disk immediately (not fully lazy)
        // but is required if HMM is not supported.
        println!("Registering MMAP pointer with HIP...");
        let status = hipHostRegister(ptr, size_bytes, hipHostRegisterDefault);
        if status != HIP_SUCCESS {
            println!("hipHostRegister failed: {:?}", status);
            // We continue to see if it works anyway (unlikely if HMM is NO)
        } else {
            println!("hipHostRegister success.");
        }

        // Ensure flushed to "disk" (page cache) somewhat?
        // Not strictly necessary for shared mapping visibility, but good practice.
        libc::msync(ptr, size_bytes, libc::MS_SYNC);

        // --- Kernel ---
        // Kernel reads from mmap_ptr and writes result to a standard output buffer
        // We do NOT copy mmap_ptr to device. We pass it DIRECTLY.
        let source_code = r#"extern "C" __global__ void reader_kernel(float* mmap_data, float* out, int n) {
    if (threadIdx.x == 0) {
        out[0] = mmap_data[0];
        out[1] = mmap_data[n / 2];
        out[2] = mmap_data[n - 1];
    }
}
"#;
        let source = CString::new(source_code).unwrap();
        let func_name = CString::new("reader_kernel").unwrap();

        // Compile
        println!("Compiling kernel...");
        let mut program: hiprtcProgram = ptr::null_mut();
        hiprtcCreateProgram(
            &mut program,
            source.as_ptr(),
            ptr::null(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        hiprtcCompileProgram(program, 0, ptr::null_mut());
        let mut code_size = 0;
        hiprtcGetCodeSize(program, &mut code_size);
        let mut code = vec![0u8; code_size];
        hiprtcGetCode(program, code.as_mut_ptr() as *mut _);
        hiprtcDestroyProgram(&mut program);

        // Load
        let mut module: hipModule_t = ptr::null_mut();
        let mut function: hipFunction_t = ptr::null_mut();
        hipModuleLoadData(&mut module, code.as_ptr() as *const _);
        hipModuleGetFunction(&mut function, module, func_name.as_ptr());

        // Output buffer (standard device memory)
        let mut d_out: *mut libc::c_void = ptr::null_mut();
        hipMalloc(&mut d_out, 3 * 4); // 3 floats

        // Args
        let n_c = n as i32;
        let mut args: [*mut libc::c_void; 3] = [
            &f32_ptr as *const _ as *mut _, // PASSING MMAP POINTER DIRECTLY
            &d_out as *const _ as *mut _,
            &n_c as *const _ as *mut _,
        ];

        // Launch
        println!("Launching kernel with MMAP pointer...");
        let status = hipModuleLaunchKernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            ptr::null_mut(),
            args.as_mut_ptr(),
            ptr::null_mut(),
        );

        if status != HIP_SUCCESS {
            println!("Launch failed with error: {:?}", status);
            println!(
                "Note: If this is 'hipErrorInvalidValue' or access violation, XNACK might be disabled or HMM unsupported for mmap."
            );
            // Try to continue to cleanup?
        } else {
            hipDeviceSynchronize();
            println!("Kernel finished.");

            // Read Result
            let mut h_out = vec![0.0f32; 3];
            hipMemcpy(
                h_out.as_mut_ptr() as *mut _,
                d_out,
                12,
                hipMemcpyKind_hipMemcpyDeviceToHost,
            );

            println!("Results: [{}, {}, {}]", h_out[0], h_out[1], h_out[2]);

            if h_out[0] == 1.0 && h_out[1] == 2.0 && h_out[2] == 3.0 {
                println!("SUCCESS: GPU read data from MMAP'd file!");
            } else {
                println!("FAILURE: Data mismatch.");
            }
        }

        // Cleanup
        hipFree(d_out);
        hipModuleUnload(module);
        libc::munmap(ptr, size_bytes);
        // std::fs::remove_file(filename)?; // keep file for inspection if needed, or remove
    }
    Ok(())
}
