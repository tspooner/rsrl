use domains::Transition;
use geometry::{Space, ActionSpace};


pub trait Agent<S: Space> {
    fn pi(&mut self, s: &S::Repr) -> usize;
    fn pi_target(&mut self, s: &S::Repr) -> usize;

    fn learn_transition(&mut self, t: &Transition<S, ActionSpace>);
}


pub mod actor_critic;
pub mod td;
