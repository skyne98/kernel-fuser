#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
use cubecl_hip_sys::*;
use std::ffi::CStr;
use std::mem;
use std::os::raw::{c_char, c_int, c_uint};

macro_rules! check {
    ($e:expr) => {{ 
        let res = $e;
        if res != HIP_SUCCESS {
            anyhow::bail!("HIP Error at line {}: {:?}", line!(), res);
        }
    }};
}

// Precise layout for ROCm 5.7 as found in /usr/include/hip/hip_runtime_api.h
#[repr(C)]
pub struct hipDeviceProp_t_v57 {
    pub name: [c_char; 256],
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: c_int,
    pub warpSize: c_int,
    pub maxThreadsPerBlock: c_int,
    pub maxThreadsDim: [c_int; 3],
    pub maxGridSize: [c_int; 3],
    pub clockRate: c_int,
    pub memoryClockRate: c_int,
    pub memoryBusWidth: c_int,
    pub totalConstMem: usize,
    pub major: c_int,
    pub minor: c_int,
    pub multiProcessorCount: c_int,
    pub l2CacheSize: c_int,
    pub maxThreadsPerMultiProcessor: c_int,
    pub computeMode: c_int,
    pub clockInstructionRate: c_int,
    pub arch: c_uint,
    pub concurrentKernels: c_int,
    pub pciDomainID: c_int,
    pub pciBusID: c_int,
    pub pciDeviceID: c_int,
    pub maxSharedMemoryPerMultiProcessor: usize,
    pub isMultiGpuBoard: c_int,
    pub canMapHostMemory: c_int,
    pub gcnArch: c_int,
    pub gcnArchName: [c_char; 256],
    pub integrated: c_int,
    pub cooperativeLaunch: c_int,
    pub cooperativeMultiDeviceLaunch: c_int,
    pub maxTexture1DLinear: c_int,
    pub maxTexture1D: c_int,
    pub maxTexture2D: [c_int; 2],
    pub maxTexture3D: [c_int; 3],
    pub hdpMemFlushCntl: *mut c_uint,
    pub hdpRegFlushCntl: *mut c_uint,
    pub memPitch: usize,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub kernelExecTimeoutEnabled: c_int,
    pub ECCEnabled: c_int,
    pub tccDriver: c_int,
    pub cooperativeMultiDeviceUnmatchedFunc: c_int,
    pub cooperativeMultiDeviceUnmatchedGridDim: c_int,
    pub cooperativeMultiDeviceUnmatchedBlockDim: c_int,
    pub cooperativeMultiDeviceUnmatchedSharedMem: c_int,
    pub isLargeBar: c_int,
    pub asicRevision: c_int,
    pub managedMemory: c_int,
    pub directManagedMemAccessFromHost: c_int,
    pub concurrentManagedAccess: c_int,
    pub pageableMemoryAccess: c_int,
    pub pageableMemoryAccessUsesHostPageTables: c_int,
}

unsafe extern "C" {
    #[link_name = "hipGetDeviceProperties"]
    pub fn hipGetDeviceProperties_v57(
        prop: *mut hipDeviceProp_t_v57,
        deviceId: c_int,
    ) -> hipError_t;
}

fn main() -> anyhow::Result<()> {
    unsafe {
        let mut count: c_int = 0;
        check!(hipGetDeviceCount(&mut count));
        println!("Found {} HIP devices\n", count);

        for i in 0..count {
            check!(hipSetDevice(i));
            
            let mut props: hipDeviceProp_t_v57 = mem::zeroed();
            check!(hipGetDeviceProperties_v57(&mut props, i));

            let name = CStr::from_ptr(props.name.as_ptr()).to_string_lossy();
            println!("---\nDevice {}: {} ---", i, name);
            println!("  Compute Capability: {}.{}", props.major, props.minor);
            println!("  GCN Arch Name:      {}", CStr::from_ptr(props.gcnArchName.as_ptr()).to_string_lossy());
            
            // --- HBM Memory ---
            let mut free: usize = 0;
            let mut total: usize = 0;
            check!(hipMemGetInfo(&mut free, &mut total));
            println!("  HBM Total:      {:>10} MB", total / 1024 / 1024);
            println!("  HBM Free:       {:>10} MB", free / 1024 / 1024);
            
            let mem_clock_mhz = props.memoryClockRate as f32 / 1000.0;
            println!("  Memory Clock:   {:>10.2} MHz", mem_clock_mhz);
            println!("  Memory Bus:     {:>10} bits", props.memoryBusWidth);
            
            // Peak Bandwidth: (Clock * 2 (DDR) * BusWidth / 8) / 1000
            let bw = (mem_clock_mhz * 2.0 * props.memoryBusWidth as f32 / 8.0) / 1000.0;
            println!("  Peak Bandwidth: {:>10.2} GB/s", bw);

            // --- Caches ---
            println!("  L2 Cache Size:  {:>10} KB", props.l2CacheSize / 1024);

            // Detailed props
            println!("  Shared Mem/Block: {:>10} KB", props.sharedMemPerBlock / 1024);
            println!("  Max Threads/Block:{:>10}", props.maxThreadsPerBlock);
            println!("  MultiProcessors:  {:>10}", props.multiProcessorCount);
            println!("  Warp Size:        {:>10}", props.warpSize);
            
            // Query attributes for cache support
            let mut l1_global: i32 = 0;
            let mut l1_local: i32 = 0;
            hipDeviceGetAttribute(&mut l1_global, hipDeviceAttribute_t_hipDeviceAttributeGlobalL1CacheSupported, i);
            hipDeviceGetAttribute(&mut l1_local, hipDeviceAttribute_t_hipDeviceAttributeLocalL1CacheSupported, i);

            println!("  Global L1 Cache:  {:>10}", if l1_global != 0 { "Supported" } else { "No" });
            println!("  Local L1 Cache:   {:>10}", if l1_local != 0 { "Supported" } else { "No" });
            
            println!("  Managed Memory:   {:>10} ", if props.managedMemory != 0 { "Yes" } else { "No" });
            println!("  Pageable Memory:  {:>10} ", if props.pageableMemoryAccess != 0 { "Yes" } else { "No" });
            println!("  HMM Host Page Tbl:{:>10} ", if props.pageableMemoryAccessUsesHostPageTables != 0 { "Yes" } else { "No" });

            println!();
        }
    }
    Ok(())
}