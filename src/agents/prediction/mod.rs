//! Prediction agents module.

use super::Agent;

pub trait Predictor<S: Sized>: Agent<Sample = (S, S, f64)> {
    fn evaluate(&self, s: &S) -> f64;
}

pub trait TDPredictor<S: Sized>: Predictor<S> {
    fn handle_td_error(&mut self, sample: &Self::Sample, td_error: f64);
    fn compute_td_error(&self, sample: &Self::Sample) -> f64;
}

pub mod gtd;
pub mod mc;
pub mod td;

// TODO:
// Implement the algorithms discussed in https://arxiv.org/pdf/1304.3999.pdf
