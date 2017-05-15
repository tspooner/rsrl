use super::Agent;
use geometry::Space;

pub trait ControlAgent<S: Space>: Agent<S> {
    fn pi(&mut self, s: &S::Repr) -> usize;
    fn pi_target(&mut self, s: &S::Repr) -> usize;
}


pub mod td;
pub mod actor_critic;
