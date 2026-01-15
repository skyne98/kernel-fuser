use cubecl_hip_sys::*;
use crate::hip::error::{check_hip, Result};
use std::marker::PhantomData;
use std::os::raw::c_void;
use std::ptr;

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
    #[allow(dead_code)]
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
