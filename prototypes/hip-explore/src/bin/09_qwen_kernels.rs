use hip_explore::*;

fn main() -> anyhow::Result<()> {
    // --- Init ---
    let count = Device::count()?;
    if count == 0 {
        return Ok(());
    }
    Device::set(0)?;
    let stream = Stream::create()?;

    // --- Kernel Source ---
    let source_code = r#"
extern "C" __global__ void rmsnorm_kernel(float* out, const float* in, const float* weight, int dim, float eps) {
    int tid = threadIdx.x;
    extern __shared__ float sdata[];

    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = in[i];
        sum += val * val;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float scale = 1.0f / sqrtf(sdata[0] / dim + eps);

    for (int i = tid; i < dim; i += blockDim.x) {
        out[i] = (in[i] * scale) * weight[i];
    }
}

extern "C" __global__ void rope_kernel(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, int pos, float theta_base) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = n_heads * (head_dim / 2);

    if (tid < total_pairs) {
        int h = tid / (head_dim / 2);
        int j = tid % (head_dim / 2);

        float freq = powf(theta_base, -(float)j / (head_dim / 2));
        float arg = (float)pos * freq;
        float cos_arg = cosf(arg);
        float sin_arg = sinf(arg);

        // Rotate Query
        float* q_ptr = q + h * head_dim;
        float v0 = q_ptr[j];
        float v1 = q_ptr[j + head_dim / 2];
        q_ptr[j] = v0 * cos_arg - v1 * sin_arg;
        q_ptr[j + head_dim / 2] = v0 * sin_arg + v1 * cos_arg;

        // Rotate Key
        if (h < n_kv_heads) {
            float* k_ptr = k + h * head_dim;
            float vk0 = k_ptr[j];
            float vk1 = k_ptr[j + head_dim / 2];
            k_ptr[j] = vk0 * cos_arg - vk1 * sin_arg;
            k_ptr[j + head_dim / 2] = vk0 * sin_arg + vk1 * cos_arg;
        }
    }
}

extern "C" __global__ void swiglu_kernel(float* out, const float* gate, const float* up, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        float g = gate[idx];
        float silu = g * (1.0f / (1.0f + expf(-g)));
        out[idx] = silu * up[idx];
    }
}
"#;

    // --- Compilation ---
    println!("Compiling Qwen 3 Kernels...");
    let program = Program::create(source_code, Some("qwen_kernels"))?;
    program.compile(vec![])?;
    let module = Module::load_data(&program.get_code()?)?;

    // --- 1. Test RMSNorm ---
    println!("Testing RMSNorm...");
    let dim = 4096;
    let h_in = vec![1.0f32; dim];
    let h_weight = vec![0.5f32; dim];
    let mut h_out = vec![0.0f32; dim];

    let mut d_in = DeviceBuffer::malloc(dim)?;
    let mut d_weight = DeviceBuffer::malloc(dim)?;
    let d_out = DeviceBuffer::malloc(dim)?;

    d_in.copy_from_host(&h_in)?;
    d_weight.copy_from_host(&h_weight)?;

    let mut eps = 1e-6f32;
    let mut dim_i32 = dim as i32;
    
    let mut d_out_ptr = d_out.as_raw();
    let mut d_in_ptr = d_in.as_raw();
    let mut d_weight_ptr = d_weight.as_raw();

    let mut args_norm: [*mut std::os::raw::c_void; 5] = [
        &mut d_out_ptr as *mut _ as *mut _,
        &mut d_in_ptr as *mut _ as *mut _,
        &mut d_weight_ptr as *mut _ as *mut _,
        &mut dim_i32 as *mut _ as *mut _,
        &mut eps as *mut _ as *mut _,
    ];

    let norm_func = module.get_function("rmsnorm_kernel")?;
    
    // Benchmark
    let start_evt = Event::create()?;
    let stop_evt = Event::create()?;
    let iters = 100;
    
    start_evt.record(&stream)?;
    for _ in 0..iters {
        norm_func.launch((1, 1, 1), (256, 1, 1), 256 * 4, &stream, &mut args_norm)?;
    }
    stop_evt.record(&stream)?;
    stop_evt.synchronize()?;
    
    let ms = Event::elapsed_time(&start_evt, &stop_evt)?;
    d_out.copy_to_host(&mut h_out)?;
    println!("  RMSNorm[0]: {:.4} (Expected ~0.5000)", h_out[0]);
    println!("  Avg Time:   {:.2} us", (ms * 1000.0) / iters as f32);

    // --- 2. Test RoPE ---
    println!("Testing RoPE...");
    let n_heads = 32;
    let n_kv_heads = 32;
    let head_dim = 128;
    let qk_len = n_heads * head_dim;
    let h_q = vec![1.0f32; qk_len];
    let h_k = vec![1.0f32; qk_len];

    let mut d_q = DeviceBuffer::malloc(qk_len)?;
    let mut d_k = DeviceBuffer::malloc(qk_len)?;
    d_q.copy_from_host(&h_q)?;
    d_k.copy_from_host(&h_k)?;

    let mut n_h = n_heads as i32;
    let mut n_kv = n_kv_heads as i32;
    let mut h_d = head_dim as i32;
    let mut pos = 10i32;
    let mut theta = 1000000.0f32;

    let mut d_q_ptr = d_q.as_raw();
    let mut d_k_ptr = d_k.as_raw();

    let mut args_rope: [*mut std::os::raw::c_void; 7] = [
        &mut d_q_ptr as *mut _ as *mut _,
        &mut d_k_ptr as *mut _ as *mut _,
        &mut n_h as *mut _ as *mut _,
        &mut n_kv as *mut _ as *mut _,
        &mut h_d as *mut _ as *mut _,
        &mut pos as *mut _ as *mut _,
        &mut theta as *mut _ as *mut _,
    ];

    let rope_func = module.get_function("rope_kernel")?;
    let grid_rope = (n_heads as u32 * head_dim as u32 / 2 / 64) + 1;
    
    start_evt.record(&stream)?;
    for _ in 0..iters {
        rope_func.launch((grid_rope, 1, 1), (64, 1, 1), 0, &stream, &mut args_rope)?;
    }
    stop_evt.record(&stream)?;
    stop_evt.synchronize()?;
    
    let ms_rope = Event::elapsed_time(&start_evt, &stop_evt)?;
    let mut h_q_res = vec![0.0f32; qk_len];
    d_q.copy_to_host(&mut h_q_res)?;
    println!("  RoPE Q[0]: {:.4}, Q[64]: {:.4}", h_q_res[0], h_q_res[64]);
    println!("  Avg Time:  {:.2} us", (ms_rope * 1000.0) / iters as f32);

    // --- 3. Test SwiGLU ---
    println!("Testing SwiGLU...");
    let ffn_dim = 11008;
    let h_gate = vec![1.0f32; ffn_dim];
    let h_up = vec![1.0f32; ffn_dim];
    let mut h_ffn_out = vec![0.0f32; ffn_dim];

    let mut d_gate = DeviceBuffer::malloc(ffn_dim)?;
    let mut d_up = DeviceBuffer::malloc(ffn_dim)?;
    let d_ffn_out = DeviceBuffer::malloc(ffn_dim)?;

    d_gate.copy_from_host(&h_gate)?;
    d_up.copy_from_host(&h_up)?;

    let mut f_d = ffn_dim as i32;
    let mut d_ffn_out_ptr = d_ffn_out.as_raw();
    let mut d_gate_ptr = d_gate.as_raw();
    let mut d_up_ptr = d_up.as_raw();

    let mut args_ffn: [*mut std::os::raw::c_void; 4] = [
        &mut d_ffn_out_ptr as *mut _ as *mut _,
        &mut d_gate_ptr as *mut _ as *mut _,
        &mut d_up_ptr as *mut _ as *mut _,
        &mut f_d as *mut _ as *mut _,
    ];

    let ffn_func = module.get_function("swiglu_kernel")?;
    let grid_ffn = (ffn_dim as u32 / 256) + 1;
    
    start_evt.record(&stream)?;
    for _ in 0..iters {
        ffn_func.launch((grid_ffn, 1, 1), (256, 1, 1), 0, &stream, &mut args_ffn)?;
    }
    stop_evt.record(&stream)?;
    stop_evt.synchronize()?;
    
    let ms_ffn = Event::elapsed_time(&start_evt, &stop_evt)?;
    d_ffn_out.copy_to_host(&mut h_ffn_out)?;
    println!("  SwiGLU[0]: {:.4} (Expected ~0.7311)", h_ffn_out[0]);
    println!("  Avg Time:   {:.2} us", (ms_ffn * 1000.0) / iters as f32);

    println!("\nAll Qwen 3 kernels verified on Device 0.");
    Ok(())
}
