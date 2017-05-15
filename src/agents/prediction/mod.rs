use super::Agent;
use geometry::{Space, NullSpace};


pub trait PredictionAgent<S: Space>: Agent<S> {
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64);
}


pub mod td;
