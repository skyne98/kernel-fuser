use cubecl_hip_sys::*;
use crate::hip::error::{check_rtc, check_hip, Result, Error};
use crate::hip::stream::Stream;
use std::ffi::{CStr, CString};
use std::os::raw::{c_int, c_void};
use std::ptr;

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
                stream.as_raw(),
                args.as_mut_ptr(),
                ptr::null_mut(),
            ))?
        };
        Ok(())
    }
}
