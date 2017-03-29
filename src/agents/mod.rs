use domain::Transition;
use geometry::{Space, ActionSpace};


pub trait Agent<S: Space> {
    fn handle(&mut self, t: &Transition<S, ActionSpace>) -> usize;
}


mod td_zero;
pub use self::td_zero::{QLearning, SARSA};
