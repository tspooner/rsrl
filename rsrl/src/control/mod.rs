//! Control agents module.
// Critic-only:
pub mod td;

// Actor-only:
pub mod mc;

// Actor-Critic:
pub mod ac;
pub mod nac;
pub mod cacla;

// TODO
// Proximal gradient-descent methods:
// https://arxiv.org/pdf/1210.4893.pdf
// https://arxiv.org/pdf/1405.6757.pdf

// TODO
// Hamid Maei Thesis (reference)
// https://era.library.ualberta.ca/files/8s45q967t/Hamid_Maei_PhDThesis.pdf
