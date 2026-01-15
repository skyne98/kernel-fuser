use cubecl_hip_sys::*;
pub use cubecl_hip_sys::{hipError_t as RawHipError, hiprtcResult as RawHipRtcError};

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
