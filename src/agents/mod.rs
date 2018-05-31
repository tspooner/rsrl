//! Learning agents module.
use domains::Transition;
use policies::Policy;
use Handler;

pub mod memory;

pub mod control;
pub mod prediction;

pub trait Controller<S, A>: Handler<Transition<S, A>> {
    /// Sample the target policy for a given state `s`.
    fn pi(&mut self, s: &S) -> A;

    /// Sample the behaviour policy for a given state `s`.
    fn mu(&mut self, s: &S) -> A;

    /// Sample a given policy against some state `s` for this agent.
    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> A;
}

pub trait Predictor<S: Sized>: Handler<Transition<S, ()>> {
    fn evaluate(&mut self, s: &S) -> f64;
}

pub trait TDPredictor<S: Sized>: Predictor<S> {
    fn handle_td_error(&mut self, sample: &Transition<S, ()>, td_error: f64);
    fn compute_td_error(&self, sample: &Transition<S, ()>) -> f64;
}

// TODO
// Proximal gradient-descent methods:
// https://arxiv.org/pdf/1210.4893.pdf
// https://arxiv.org/pdf/1405.6757.pdf

// TODO
// Hamid Maei Thesis (reference)
// https://era.library.ualberta.ca/files/8s45q967t/Hamid_Maei_PhDThesis.pdf
