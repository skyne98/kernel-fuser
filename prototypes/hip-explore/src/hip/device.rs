use cubecl_hip_sys::*;
use crate::hip::error::{check_hip, Result};

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
