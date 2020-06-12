//! Agent policies module.
//!
//! This module contains fixed and parameterised policies for reinforcement
//! learning control problems. A policy is considered to be either deterministic
//! or stochastic, for which we have the following definitions, respectively: 1)
//! _π(X) -> U_; 2) _π(X, U) -> R_, or equivalently, _π(x) -> Ω(U)_; where _X_
//! and _U_ are the state and action spaces, respectively; _R_ is the set
//! of real values; and _Ω(U)_ is the set of probability measures on the action
//! set _U_. In general, deterministic policies may be considered a special case
//! of stochastic policies in which all probability mass is placed on a single
//! action _u'_ for any given state _x_. For continuous policies, this can be
//! seen as a dirac delta distribution, _δ(u' - u)_.
use crate::{Differentiable, Enumerable, Function, OutputOf, Shared};
use ndarray::Array2;
use rand::{thread_rng, Rng};

mod greedy;
mod random;
mod epsilon_greedy;

pub use self::greedy::Greedy;
pub use self::random::Random;
pub use self::epsilon_greedy::EpsilonGreedy;

mod beta;
mod gaussian;
mod softmax;

pub use self::beta::Beta;
pub use self::gaussian::Gaussian;
pub use self::softmax::{Gibbs, Softmax};

mod ipp;
mod point;

pub use self::ipp::IPP;
pub use self::point::Point;

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
pub trait Policy<S>:
    Function<(S, <Self as Policy<S>>::Action), Output = f64>
    + for<'a> Function<(S, &'a <Self as Policy<S>>::Action), Output = f64>
{
    type Action: Sized;

    /// Sample the (possibly stochastic) policy distribution for a given
    /// `state`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, state: S) -> Self::Action;

    /// Return the most probable action according to the policy distribution,
    /// if well-defined.
    fn mode(&self, state: S) -> Self::Action;
}

impl<S, T: Policy<S>> Policy<S> for Shared<T> {
    type Action = T::Action;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, state: S) -> Self::Action {
        self.borrow().sample(rng, state)
    }

    fn mode(&self, state: S) -> Self::Action { self.borrow().mode(state) }
}

/// Trait for policies that are defined on an enumerable action space.
pub trait EnumerablePolicy<S>: Policy<S, Action = usize> + Enumerable<(S,)>
where
    OutputOf<Self, (S,)>: std::ops::Index<usize, Output = f64> + IntoIterator<Item = f64>,
    <OutputOf<Self, (S,)> as IntoIterator>::IntoIter: ExactSizeIterator,
{
}

impl<S, P> EnumerablePolicy<S> for P
where
    P: Policy<S, Action = usize> + Enumerable<(S,)>,

    OutputOf<Self, (S,)>: std::ops::Index<usize, Output = f64> + IntoIterator<Item = f64>,
    <OutputOf<Self, (S,)> as IntoIterator>::IntoIter: ExactSizeIterator,
{
}

/// Trait for policies with a representation that is differentiable wrt its
/// parameters.
pub trait DifferentiablePolicy<S>:
    Policy<S>
    + Differentiable<(S, <Self as Policy<S>>::Action), Jacobian = Array2<f64>>
    + for<'a> Differentiable<(S, &'a <Self as Policy<S>>::Action), Jacobian = Array2<f64>>
{
}

impl<S, P> DifferentiablePolicy<S> for P where P: Policy<S>
        + Differentiable<(S, <Self as Policy<S>>::Action), Jacobian = Array2<f64>>
        + for<'a> Differentiable<(S, &'a <Self as Policy<S>>::Action), Jacobian = Array2<f64>>
{
}
