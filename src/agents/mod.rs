//! Learning agents module.

pub mod memory;

pub mod control;
pub mod prediction;

pub use self::control::ControlAgent;
pub use self::prediction::PredictionAgent;


pub trait BatchAgent {
    fn consolidate(&mut self);
}


// TODO
// Proximal gradient-descent methods:
// https://arxiv.org/pdf/1210.4893.pdf
// https://arxiv.org/pdf/1405.6757.pdf

// TODO
// Hamid Maei Thesis (reference)
// https://era.library.ualberta.ca/files/8s45q967t/Hamid_Maei_PhDThesis.pdf
