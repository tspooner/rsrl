//! Prediction agents module.
use crate::Shared;

pub trait ValuePredictor<S> {
    /// Compute the estimated value of V(s).
    fn predict_v(&self, s: S) -> f64;
}

impl<S, T: ValuePredictor<S>> ValuePredictor<S> for Shared<T> {
    fn predict_v(&self, s: S) -> f64 { self.borrow().predict_v(s) }
}

pub trait ActionValuePredictor<S, A> {
    /// Compute the estimated value of Q(s, a).
    fn predict_q(&self, s: S, a: A) -> f64;
}

impl<S, A, T: ActionValuePredictor<S, A>> ActionValuePredictor<S, A> for Shared<T> {
    fn predict_q(&self, s: S, a: A) -> f64 { self.borrow().predict_q(s, a) }
}

pub mod lstd;
pub mod mc;
pub mod td;

// TODO:
// Implement the algorithms discussed in https://arxiv.org/pdf/1304.3999.pdf
