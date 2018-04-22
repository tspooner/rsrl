//! Control agents module.

use super::Agent;
use domains::Transition;

pub trait Controller<S, A>: Agent<Sample = Transition<S, A>> {
    /// Sample the target policy for a given state `s`.
    fn pi(&mut self, s: &S) -> A;

    /// Sample the behaviour policy for a given state `s`.
    fn mu(&mut self, s: &S) -> A;
}

pub mod actor_critic;
pub mod gtd;
pub mod td;
pub mod totd;
