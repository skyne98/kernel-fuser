use cubecl_hip_sys::*;
use crate::hip::error::{check_hip, Result};
use std::ptr;

pub struct Stream {
    pub(crate) inner: hipStream_t,
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
