//! Prediction agents module.
pub use crate::core::{ActionValuePredictor, ValuePredictor};

pub mod gtd;
pub mod lstd;
pub mod mc;
pub mod td;

// TODO:
// Implement the algorithms discussed in https://arxiv.org/pdf/1304.3999.pdf
