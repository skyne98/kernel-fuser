#![allow(unsafe_op_in_unsafe_fn)]
use cubecl_hip_sys::*;
pub use cubecl_hip_sys::{hipError_t as RawHipError, hiprtcResult as RawHipRtcError};
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::os::raw::{c_int, c_void};
use std::ptr;

// --- Error Handling ---

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("HIP API Error: {0:?}")]
    Hip(RawHipError),
    #[error("HIPRTC Error: {0:?}")]
    Rtc(RawHipRtcError),
    #[error("Other Error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn check_hip(res: RawHipError) -> Result<()> {
    if res == HIP_SUCCESS {
        Ok(())
    } else {
        Err(Error::Hip(res))
    }
}

pub fn check_rtc(res: RawHipRtcError) -> Result<()> {
    if res == hiprtcResult_HIPRTC_SUCCESS {
        Ok(())
    } else {
        Err(Error::Rtc(res))
    }
}

// --- Device Management ---

pub struct Device;

impl Device {
    pub fn count() -> Result<i32> {
        let mut count = 0;
        unsafe { check_hip(hipGetDeviceCount(&mut count))? };
        Ok(count)
    }

    pub fn set(id: i32) -> Result<()> {
        unsafe { check_hip(hipSetDevice(id))? };
        Ok(())
    }

    pub fn synchronize() -> Result<()> {
        unsafe { check_hip(hipDeviceSynchronize())? };
        Ok(())
    }

    pub fn memory_info() -> Result<(usize, usize)> {
        let mut free = 0;
        let mut total = 0;
        unsafe { check_hip(hipMemGetInfo(&mut free, &mut total))? };
        Ok((free, total))
    }
}

// --- Streams ---

pub struct Stream {
    inner: hipStream_t,
}

impl Stream {
    pub fn create() -> Result<Self> {
        let mut inner = ptr::null_mut();
        unsafe { check_hip(hipStreamCreate(&mut inner))? };
        Ok(Self { inner })
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { check_hip(hipStreamSynchronize(self.inner))? };
        Ok(())
    }

    pub fn as_raw(&self) -> hipStream_t {
        self.inner
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            let _ = hipStreamDestroy(self.inner);
        }
    }
}

// --- Events ---

pub struct Event {
    inner: hipEvent_t,
}

impl Event {
    pub fn create() -> Result<Self> {
        let mut inner = ptr::null_mut();
        unsafe { check_hip(hipEventCreate(&mut inner))? };
        Ok(Self { inner })
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        unsafe { check_hip(hipEventRecord(self.inner, stream.inner))? };
        Ok(())
    }

    pub fn synchronize(&self) -> Result<()> {
        unsafe { check_hip(hipEventSynchronize(self.inner))? };
        Ok(())
    }

    pub fn elapsed_time(start: &Event, end: &Event) -> Result<f32> {
        let mut ms = 0.0;
        unsafe { check_hip(hipEventElapsedTime(&mut ms, start.inner, end.inner))? };
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            let _ = hipEventDestroy(self.inner);
        }
    }
}

// --- Memory Management ---

pub struct DeviceBuffer<T> {
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    pub fn malloc(len: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        let size = len * std::mem::size_of::<T>();
        unsafe { check_hip(hipMalloc(&mut ptr, size))? };
        Ok(Self {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> Result<()> {
        assert!(src.len() <= self.len);
        let size = src.len() * std::mem::size_of::<T>();
        unsafe {
            check_hip(hipMemcpy(
                self.ptr,
                src.as_ptr() as *const c_void,
                size,
                hipMemcpyKind_hipMemcpyHostToDevice,
            ))?
        };
        Ok(())
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        assert!(dst.len() <= self.len);
        let size = dst.len() * std::mem::size_of::<T>();
        unsafe {
            check_hip(hipMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                self.ptr,
                size,
                hipMemcpyKind_hipMemcpyDeviceToHost,
            ))?
        };
        Ok(())
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = hipFree(self.ptr);
        }
    }
}

pub struct ManagedBuffer<T> {
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> ManagedBuffer<T> {
    pub fn malloc(len: usize) -> Result<Self> {
        let mut ptr = ptr::null_mut();
        let size = len * std::mem::size_of::<T>();
        unsafe { check_hip(hipMallocManaged(&mut ptr, size, hipMemAttachGlobal))? };
        Ok(Self {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    pub fn as_raw(&self) -> *mut c_void {
        self.ptr
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) }
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
    }
}

impl<T> Drop for ManagedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = hipFree(self.ptr);
        }
    }
}

pub struct HostRegistration {
    ptr: *mut c_void,
    size: usize,
}

impl HostRegistration {
    pub fn register<T>(data: &mut [T]) -> Result<Self> {
        let ptr = data.as_mut_ptr() as *mut c_void;
        let size = data.len() * std::mem::size_of::<T>();
        unsafe { check_hip(hipHostRegister(ptr, size, hipHostRegisterDefault))? };
        Ok(Self { ptr, size })
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        unsafe {
            let _ = hipHostUnregister(self.ptr);
        }
    }
}

// --- HIPRTC ---

pub struct Program {
    inner: hiprtcProgram,
}

impl Program {
    pub fn create(source: &str, name: Option<&str>) -> Result<Self> {
        let c_source = CString::new(source).map_err(|e| Error::Other(e.to_string()))?;
        let c_name = name
            .map(CString::new)
            .transpose()
            .map_err(|e| Error::Other(e.to_string()))?;
        let mut inner = ptr::null_mut();

        unsafe {
            check_rtc(hiprtcCreateProgram(
                &mut inner,
                c_source.as_ptr(),
                c_name.map_or(ptr::null(), |n| n.as_ptr()),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            ))?
        };
        Ok(Self { inner })
    }

    pub fn compile(&self, options: Vec<&str>) -> Result<()> {
        let c_options: Vec<CString> = options
            .into_iter()
            .map(|s| CString::new(s).unwrap())
            .collect();
        let ptr_options: Vec<*const i8> = c_options.iter().map(|s| s.as_ptr()).collect();

        let status = unsafe {
            hiprtcCompileProgram(
                self.inner,
                ptr_options.len() as c_int,
                ptr_options.as_ptr() as *mut *const i8,
            )
        };

        if status != hiprtcResult_HIPRTC_SUCCESS {
            let mut log_size = 0;
            unsafe { hiprtcGetProgramLogSize(self.inner, &mut log_size) };
            let mut log = vec![0i8; log_size];
            unsafe { hiprtcGetProgramLog(self.inner, log.as_mut_ptr()) };
            let log_str = unsafe { CStr::from_ptr(log.as_ptr()).to_string_lossy().into_owned() };
            return Err(Error::Other(format!("Compilation failed: {}", log_str)));
        }
        Ok(())
    }

    pub fn get_code(&self) -> Result<Vec<u8>> {
        let mut size = 0;
        unsafe { check_rtc(hiprtcGetCodeSize(self.inner, &mut size))? };
        let mut code = vec![0u8; size];
        unsafe { check_rtc(hiprtcGetCode(self.inner, code.as_mut_ptr() as *mut _))? };
        Ok(code)
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe {
            let _ = hiprtcDestroyProgram(&mut self.inner);
        }
    }
}

pub struct Module {
    inner: hipModule_t,
}

impl Module {
    pub fn load_data(code: &[u8]) -> Result<Self> {
        let mut inner = ptr::null_mut();
        unsafe { check_hip(hipModuleLoadData(&mut inner, code.as_ptr() as *const _))? };
        Ok(Self { inner })
    }

    pub fn get_function(&self, name: &str) -> Result<Function> {
        let c_name = CString::new(name).map_err(|e| Error::Other(e.to_string()))?;
        let mut func = ptr::null_mut();
        unsafe { check_hip(hipModuleGetFunction(&mut func, self.inner, c_name.as_ptr()))? };
        Ok(Function { inner: func })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            let _ = hipModuleUnload(self.inner);
        }
    }
}

pub struct Function {
    inner: hipFunction_t,
}

impl Function {
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: &Stream,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        unsafe {
            check_hip(hipModuleLaunchKernel(
                self.inner,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem,
                stream.inner,
                args.as_mut_ptr(),
                ptr::null_mut(),
            ))?
        };
        Ok(())
    }
}
