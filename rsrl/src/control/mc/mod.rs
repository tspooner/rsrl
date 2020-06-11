//! Monte-Carlo policy gradient algorithms.
pub mod reinforce;
pub mod baseline_reinforce;

pub use self::{
    reinforce::REINFORCE,
    baseline_reinforce::BaselineREINFORCE,
};
