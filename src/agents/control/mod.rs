use domains::Transition;
use geometry::Space;
use policies::Policy;

pub trait ControlAgent<S: Space, A: Space> {
    fn pi(&mut self, s: &S::Repr) -> usize;
    fn evaluate_policy(&self, p: &mut Policy, s: &S::Repr) -> usize;

    fn handle_transition(&mut self, t: &Transition<S, A>);
    fn handle_terminal(&mut self, s: &S::Repr);
}


pub mod td;
pub mod gtd;
pub mod mstd;
pub mod actor_critic;
