use domains::Transition;
use geometry::{Space, ActionSpace};


pub trait Agent<S: Space> {
    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>);
    fn handle_terminal(&mut self) {}
}

pub use self::control::ControlAgent;


pub mod control;
pub mod prediction;
