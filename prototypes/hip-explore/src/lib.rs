pub mod hip {
    pub mod device;
    pub mod error;
    pub mod event;
    pub mod memory;
    pub mod rtc;
    pub mod stream;
}

pub use hip::device::Device;
pub use hip::error::{Error, Result};
pub use hip::event::Event;
pub use hip::memory::{DeviceBuffer, ManagedBuffer, HostRegistration};
pub use hip::rtc::{Program, Module, Function};
pub use hip::stream::Stream;
