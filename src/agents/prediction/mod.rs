//! Prediction agents module.

use geometry::Space;


pub trait PredictionAgent<S: Space> {
    fn evaluate(&self, s: &S::Repr) -> f64;

    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64>;
    fn handle_terminal(&mut self, s: &S::Repr);
}


pub mod mc;
pub mod td;
pub mod gtd;

// TODO:
// Implement the algorithms discussed in https://arxiv.org/pdf/1304.3999.pdf
pub mod lstd;
