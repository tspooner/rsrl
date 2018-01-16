//! Control agents module.

use super::Agent;
use domains::Transition;
use geometry::Space;
use policies::Policy;


pub trait Controller<S: Space, A: Space>: Agent<S, Sample=Transition<S, A>> {
    /// Sample the target policy for a given state `s`.
    fn pi(&mut self, s: &S::Repr) -> usize;

    /// Sample the behaviour policy for a given state `s`.
    fn mu(&mut self, s: &S::Repr) -> usize;

    /// Sample a given policy against some state `s` for this agent.
    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize;
}


pub mod td;
pub mod gtd;
pub mod totd;
pub mod actor_critic;
