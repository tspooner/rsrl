use {Function, Parameterised};
use domain::Transition;
use geometry::{Space, ActionSpace};
use policies::Policy;


pub trait Agent<S: Space> {
    fn pi(&mut self, s: &S::Repr) -> usize;
    fn train(&mut self, t: &Transition<S, ActionSpace>);
}


pub mod actor_critic;
pub mod td;
