//! Prediction agents module.

use super::Agent;
use geometry::Space;

pub trait Predictor<S: Space>: Agent<Sample = (S::Repr, S::Repr, f64)> {
    fn evaluate(&self, s: &S::Repr) -> f64;
}

pub trait TDPredictor<S: Space>: Predictor<S> {
    fn handle_td_error(&mut self, sample: &Self::Sample, td_error: f64);
    fn compute_td_error(&self, sample: &Self::Sample) -> f64;
}

pub mod mc;
pub mod td;
pub mod gtd;

// TODO:
// Implement the algorithms discussed in https://arxiv.org/pdf/1304.3999.pdf
