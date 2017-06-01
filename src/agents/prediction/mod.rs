use geometry::Space;


pub trait PredictionAgent<S: Space> {
    fn handle_transition(&mut self, s: &S::Repr, ns: &S::Repr, r: f64) -> Option<f64>;
    fn handle_terminal(&mut self, s: &S::Repr);
}


pub mod mc;
pub mod td;
pub mod gtd;
