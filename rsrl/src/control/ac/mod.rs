//! Actor-critic algorithms.
pub mod a2c;
pub mod cacla;
pub mod nac;
pub mod offpac;
pub mod qac;
pub mod tdac;

pub use self::{a2c::A2C, cacla::CACLA, nac::NAC, offpac::OffPAC, qac::QAC, tdac::TDAC};
