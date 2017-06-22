use domains::Transition;
use geometry::Space;
use policies::Policy;

pub trait ControlAgent<S: Space, A: Space> {
    fn pi(&mut self, s: &S::Repr) -> usize;
    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize;

    fn handle_transition(&mut self, t: &Transition<S, A>);
    fn handle_terminal(&mut self, s: &S::Repr);
}


pub mod td;
pub mod gtd;
pub mod actor_critic;
