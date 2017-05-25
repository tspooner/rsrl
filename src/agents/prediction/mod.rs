use geometry::{Space, NullSpace};


pub trait PredictionAgent<S: Space> {
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> f64;
    fn handle_terminal(&mut self, s: &S::Repr);
}


pub mod td;
