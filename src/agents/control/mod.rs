use super::Agent;
use domains::Transition;
use geometry::{Space, ActionSpace};

pub trait ControlAgent<S: Space, A: Space>: Agent<S> {
    fn pi(&mut self, s: &S::Repr) -> usize;
    fn pi_target(&mut self, s: &S::Repr) -> usize;

    fn handle_transition(&mut self, t: &Transition<S, A>);
}


pub mod td;
pub mod actor_critic;
