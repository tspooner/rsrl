//! Prediction agents module.
pub use core::Predictor;

pub mod gtd;
pub mod mc;
pub mod td;
pub mod lstd;

// TODO:
// Implement the algorithms discussed in https://arxiv.org/pdf/1304.3999.pdf
