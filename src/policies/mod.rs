//! Agent policy module.
use crate::core::*;
use crate::domains::Transition;
use crate::fa::Parameterised;
use rand::Rng;

#[inline]
pub(self) fn sample_probs<R: Rng + ?Sized>(rng: &mut R, probabilities: &[f64]) -> usize {
    let r = rng.gen::<f64>();
    let n_actions = probabilities.len();

    match probabilities.into_iter().position(|p| *p > r) {
        Some(index) => index,
        None => n_actions - 1,
    }
}

/// Policy trait for functions that select between a set of values.
pub trait Policy<S>: Algorithm {
    type Action;

    /// Sample the policy distribution for a given input.
    fn sample(&mut self, input: &S) -> Self::Action;

    /// Return the probability of selecting an action for a given input.
    fn probability(&mut self, input: &S, a: Self::Action) -> f64;
}

pub trait FinitePolicy<S>: Policy<S, Action = usize> {
    /// Return the probability of selecting each action for a given input.
    fn probabilities(&mut self, input: &S) -> Vector<f64>;
}

pub trait DifferentiablePolicy<S>: Policy<S> {
    /// Compute the derivative of the log probability for a single action.
    fn grad_log(&self, input: &S, a: Self::Action) -> Matrix<f64>;
}

pub trait ParameterisedPolicy<S>: Policy<S> + Parameterised {
    fn update(&mut self, input: &S, a: Self::Action, error: f64);
    fn update_raw(&mut self, errors: Matrix<f64>);
}

pub mod fixed;
pub mod parameterised;
