use hip_explore::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // --- Discovery ---
    let count = Device::count()?;
    println!("Found {} HIP devices (via safe wrapper).", count);

    if count == 0 {
        return Ok(());
    }

    Device::set(0)?;
    let (free, total) = Device::memory_info()?;
    println!(
        "Device Memory: Free: {} MB | Total: {} MB",
        free / 1024 / 1024,
        total / 1024 / 1024
    );

    // --- Kernel Source ---
    let source_code = r#"extern "C" __global__ void kernel(float a, float *x, float *b, float *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out[tid] = x[tid] * a + b[tid];
  }
}
"#;

    // --- Compile & Load ---
    let program = Program::create(source_code, Some("simple_add"))?;
    program.compile(vec![])?;
    let code = program.get_code()?;

    let module = Module::load_data(&code)?;
    let kernel = module.get_function("kernel")?;

    // --- Prepare Data ---
    let n: i32 = 1024;
    let a: f32 = 2.0;
    let h_x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let h_b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let mut h_out = vec![0.0f32; n as usize];

    let mut d_x = DeviceBuffer::<f32>::malloc(n as usize)?;
    let mut d_b = DeviceBuffer::<f32>::malloc(n as usize)?;
    let d_out = DeviceBuffer::<f32>::malloc(n as usize)?;

    d_x.copy_from_host(&h_x)?;
    d_b.copy_from_host(&h_b)?;

    // --- Launch ---
    let stream = Stream::create()?;

    // We need to pass pointers to the arguments
    let mut d_x_ptr = d_x.as_raw();
    let mut d_b_ptr = d_b.as_raw();
    let mut d_out_ptr = d_out.as_raw();
    let mut a_val = a;
    let mut n_val = n;

    let mut args: [*mut std::os::raw::c_void; 5] = [
        &mut a_val as *mut f32 as *mut _,
        &mut d_x_ptr as *mut *mut _ as *mut _,
        &mut d_b_ptr as *mut *mut _ as *mut _,
        &mut d_out_ptr as *mut *mut _ as *mut _,
        &mut n_val as *mut i32 as *mut _,
    ];

    let block_dim = 64;
    let grid_dim = (n as u32 + block_dim - 1) / block_dim;

    let start = Instant::now();
    kernel.launch((grid_dim, 1, 1), (block_dim, 1, 1), 0, &stream, &mut args)?;
    stream.synchronize()?;
    let duration = start.elapsed();
    println!("Kernel finished in {}µs", duration.as_micros());

    // --- Verify ---
    d_out.copy_to_host(&mut h_out)?;

    let mut errors = 0;
    for i in 0..n as usize {
        let expected = h_x[i] * a + h_b[i];
        if (h_out[i] - expected).abs() > 1e-5 {
            errors += 1;
        }
    }

    if errors == 0 {
        println!("SUCCESS: Results match (verified via safe wrapper)!");
    } else {
        println!("FAILURE: {} errors found.", errors);
    }

    Ok(())
}
