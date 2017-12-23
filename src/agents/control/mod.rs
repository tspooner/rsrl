use domains::Transition;
use geometry::Space;
use policies::Policy;

pub trait ControlAgent<S: Space, A: Space> {
    /// Handle a transition between one state, `s`, and another, `s'`, given an action, `a`.
    fn handle_transition(&mut self, t: &Transition<S, A>);

    /// Handle the terminal state of an episode.
    fn handle_terminal(&mut self, s: &S::Repr);

    /// Sample the target policy for a given state `s`.
    fn pi(&mut self, s: &S::Repr) -> usize;

    /// Sample the behaviour policy for a given state `s`.
    fn mu(&mut self, s: &S::Repr) -> usize;

    /// Sample a given policy against some state `s` for this agent.
    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize;
}


pub mod td;
pub mod gtd;
pub mod actor_critic;
