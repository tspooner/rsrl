use domains::Transition;
use geometry::{Space, ActionSpace};

pub trait ControlAgent<S: Space, A: Space> {
    fn pi(&mut self, s: &S::Repr) -> usize;
    fn pi_target(&mut self, s: &S::Repr) -> usize;

    fn handle_transition(&mut self, t: &Transition<S, A>);
    fn handle_terminal(&mut self, s: &S::Repr);
}


pub mod td;
pub mod actor_critic;
