//! Monte-Carlo policy gradient algorithms.
pub mod baseline_reinforce;
pub mod reinforce;

pub use self::{baseline_reinforce::BaselineREINFORCE, reinforce::REINFORCE};
