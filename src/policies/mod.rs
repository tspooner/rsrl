//! Agent policy module.
use domains::Transition;
use geometry::{Vector, Matrix};
use rand::{Rng, ThreadRng};

#[inline]
pub(self) fn sample_probs(rng: &mut ThreadRng, probabilities: &[f64]) -> usize {
    let r = rng.next_f64();
    let n_actions = probabilities.len();

    match probabilities.into_iter().position(|p| *p > r) {
        Some(index) => index,
        None => n_actions - 1,
    }
}

/// Policy trait for functions that select between a set of values.
pub trait Policy<S> {
    type Action;

    /// Sample the policy distribution for a given input.
    fn sample(&mut self, input: &S) -> Self::Action;

    /// Return the probability of selecting an action for a given input.
    fn probability(&mut self, input: &S, a: Self::Action) -> f64;

    fn handle_terminal(&mut self, sample: &Transition<S, Self::Action>) {}
}

pub trait FinitePolicy<S>: Policy<S, Action = usize> {
    /// Return the probability of selecting each action for a given input.
    fn probabilities(&mut self, input: &S) -> Vector<f64>;
}

pub trait DifferentiablePolicy<S>: Policy<S> {
    /// Compute the derivative of the log probability for a single action.
    fn grad_log(&self, input: &S, a: Self::Action) -> Matrix<f64>;
}

pub trait ParameterisedPolicy<S>: DifferentiablePolicy<S> {
    fn update(&mut self, input: &S, a: Self::Action, error: f64) {
        let grad_log = self.grad_log(input, a);

        self.update_raw(error*grad_log)
    }

    fn update_raw(&mut self, errors: Matrix<f64>);
}

pub mod fixed;
pub mod parametrised;
