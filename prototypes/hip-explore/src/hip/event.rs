use cubecl_hip_sys::*;
use crate::hip::error::{check_hip, Result};
use crate::hip::stream::Stream;
use std::ptr;

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
        unsafe { check_hip(hipEventRecord(self.inner, stream.as_raw()))? };
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
