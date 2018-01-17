//! Prediction agents module.

use super::Agent;
use geometry::Space;


pub trait Predictor<S: Space>: Agent<S, Sample=(S::Repr, S::Repr, f64)> {
    fn evaluate(&self, s: &S::Repr) -> f64;
}

pub trait TDPredictor<S: Space>: Predictor<S> {
    fn compute_error(&self, sample: &Self::Sample) -> f64;
    fn handle_error(&mut self, sample: &Self::Sample, td_error: f64);
}


pub mod mc;
pub mod td;
pub mod gtd;

// TODO:
// Implement the algorithms discussed in https://arxiv.org/pdf/1304.3999.pdf
pub mod lstd;
pub mod lspe;
