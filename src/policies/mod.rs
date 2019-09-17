//! Agent policy module.
//!
//! This module contains [fixed](fixed/index.html) and [parameterised](parameterised/index.html)
//! policies for reinforcement learning control problems. A policy is considered to be either
//! deterministic or stochastic, for which we have the following definitions, respectively: 1)
//! _π(X) -> U_; 2) _π(X, U) -> R_, or equivalently, _π(x) -> Ω(U)_; where _X_ and _U_ are the
//! state and action spaces, respectively; _R_ is the set of real values; and _Ω(U)_ is the set of
//! probability measures on the action set _U_. In general, deterministic policies may be
//! considered a special case of stochastic policies in which all probability mass is placed on a
//! single action _u'_ for any given state _x_. For continuous policies, this can be seen as a
//! dirac delta distribution, _δ(u' - u)_.
use crate::{Algorithm, fa::Parameterised};
use ndarray::{Array2, ArrayView2};
use rand::{thread_rng, Rng};
use std::ops::AddAssign;

pub mod gaussian;

import_all!(random);
import_all!(greedy);
import_all!(epsilon_greedy);
import_all!(softmax);
import_all!(beta);
import_all!(dirichlet);
import_all!(gamma);

import_all!(ipp);
import_all!(shared);
// import_all!(perturbation);

#[allow(dead_code)]
#[inline]
pub(self) fn sample_probs(probabilities: &[f64]) -> usize {
    sample_probs_with_rng(&mut thread_rng(), probabilities)
}

#[inline]
pub(self) fn sample_probs_with_rng<R: Rng + ?Sized>(rng: &mut R, probabilities: &[f64]) -> usize {
    let r = rng.gen::<f64>();

    match probabilities
        .into_iter()
        .scan(0.0, |state, &p| {
            *state = *state + p;

            Some(*state)
        })
        .position(|p| p > r)
    {
        Some(index) => index,
        None => probabilities.len() - 1,
    }
}

/// Policy trait for functions that define a probability distribution over
/// actions.
pub trait Policy<S>: Algorithm {
    type Action;

    /// Sample the (possibly stochastic) policy distribution for a given
    /// `state`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, state: &S) -> Self::Action { self.mpa(state) }

    /// Return the "most probable action" according to the policy distribution,
    /// if well-defined.
    fn mpa(&self, _: &S) -> Self::Action { unimplemented!() }

    /// Return the probability of selecting an action for a given `state`.
    fn probability(&self, state: &S, a: &Self::Action) -> f64;
}

/// Trait for policies that are defined on a finite action space.
pub trait FinitePolicy<S>: Policy<S, Action = usize> {
    /// Return the number of actions available to the policy.
    fn n_actions(&self) -> usize;

    /// Return the probability of selecting each action for a given `state`.
    fn probabilities(&self, state: &S) -> Vec<f64>;
}

/// Trait for policies that have a representation that is differentiable wrt its parameters.
pub trait DifferentiablePolicy<S>: Policy<S> + Parameterised {
    /// Update the weights in the direction of an error for a given state and
    /// action.
    fn update(&mut self, state: &S, a: &Self::Action, error: f64);

    fn update_grad(&mut self, grad: &ArrayView2<f64>) {
        self.weights_view_mut().add_assign(grad);
    }

    fn update_grad_scaled(&mut self, grad: &ArrayView2<f64>, factor: f64) {
        self.weights_view_mut().scaled_add(factor, grad);
    }

    /// Compute the gradient of the log probability wrt the policy weights.
    fn grad(&self, state: &S, a: &Self::Action) -> Array2<f64> {
        let p = self.probability(state, a);

        self.grad_log(state, a) / p
    }

    /// Compute the gradient of the log probability wrt the policy weights.
    fn grad_log(&self, state: &S, a: &Self::Action) -> Array2<f64> {
        let p = self.probability(state, a);

        self.grad(state, a) * p
    }
}
