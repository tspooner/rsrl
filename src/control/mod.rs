//! Control agents module.
pub use crate::core::Controller;

pub mod actor_critic;
pub mod gtd;
pub mod mc;
pub mod td;
pub mod totd;

// TODO
// Proximal gradient-descent methods:
// https://arxiv.org/pdf/1210.4893.pdf
// https://arxiv.org/pdf/1405.6757.pdf

// TODO
// Hamid Maei Thesis (reference)
// https://era.library.ualberta.ca/files/8s45q967t/Hamid_Maei_PhDThesis.pdf
