//! Learning agents module.

pub mod memory;

pub mod control;
pub mod prediction;

use geometry::Space;
pub use self::control::Controller;
pub use self::prediction::{Predictor, TDPredictor};


pub trait Agent<S: Space> {
    type Sample;

    fn handle_sample(&mut self, sample: &Self::Sample);
    fn handle_terminal(&mut self, s: &S::Repr);
}


pub trait BatchAgent<S: Space>: Agent<S> {
    fn consolidate(&mut self);

    fn handle_batch(&mut self, batch: &Vec<Self::Sample>) {
        for sample in batch.iter() {
            self.handle_sample(sample);
        }
    }
}


// TODO
// Proximal gradient-descent methods:
// https://arxiv.org/pdf/1210.4893.pdf
// https://arxiv.org/pdf/1405.6757.pdf

// TODO
// Hamid Maei Thesis (reference)
// https://era.library.ualberta.ca/files/8s45q967t/Hamid_Maei_PhDThesis.pdf
