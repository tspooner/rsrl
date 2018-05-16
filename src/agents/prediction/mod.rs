//! Prediction agents module.
use domains::Transition;
use Handler;

pub trait Predictor<S: Sized>: Handler<Transition<S, ()>> {
    fn evaluate(&self, s: &S) -> f64;
}

pub trait TDPredictor<S: Sized>: Predictor<S> {
    fn handle_td_error(&mut self, sample: &Transition<S, ()>, td_error: f64);
    fn compute_td_error(&self, sample: &Transition<S, ()>) -> f64;
}

pub mod gtd;
pub mod mc;
pub mod td;

// TODO:
// Implement the algorithms discussed in https://arxiv.org/pdf/1304.3999.pdf
