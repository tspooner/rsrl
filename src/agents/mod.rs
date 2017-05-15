use domains::Transition;
use geometry::{Space, ActionSpace};


pub trait Agent<S: Space> {
    fn pi(&mut self, s: &S::Repr) -> usize;
    fn pi_target(&mut self, s: &S::Repr) -> usize;

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>);
    fn handle_terminal(&mut self) {}
}


pub mod control;
pub mod prediction;
