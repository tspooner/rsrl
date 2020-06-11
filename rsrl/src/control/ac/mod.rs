//! Actor-critic algorithms.
pub mod cacla;
pub mod qac;
pub mod tdac;
pub mod a2c;
pub mod nac;
pub mod offpac;

pub use self::{
    cacla::CACLA,
    qac::QAC,
    tdac::TDAC,
    a2c::A2C,
    nac::NAC,
    offpac::OffPAC,
};
