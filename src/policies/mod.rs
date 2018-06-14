//! Agent policy module.
use core::{Handler, Shared};
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

pub type SharedPolicy<S, A> = Shared<Policy<S, A>>;

/// Policy trait for functions that select between a set of values.
pub trait Policy<S, A>: Handler<Transition<S, A>> {
    /// Sample the policy distribution for a given input.
    fn sample(&mut self, input: &S) -> A;

    /// Return the probability of selecting an action for a given input.
    fn probability(&mut self, input: &S, a: A) -> f64;
}

pub trait FinitePolicy<S>: Policy<S, usize> {
    /// Return the probability of selecting each action for a given input.
    fn probabilities(&mut self, input: &S) -> Vector<f64>;
}

pub trait DifferentiablePolicy<S, A>: Policy<S, A> {
    /// Compute the derivative of the log probability for a single action.
    fn grad_log(&self, input: &S, a: A) -> Matrix<f64>;
}

pub trait ParameterisedPolicy<S, A>: DifferentiablePolicy<S, A> {
    fn update(&mut self, input: &S, a: A, error: f64) {
        let grad_log = self.grad_log(input, a);

        self.update_raw(error*grad_log)
    }

    fn update_raw(&mut self, errors: Matrix<f64>);
}

pub mod fixed;
pub mod parametrised;
