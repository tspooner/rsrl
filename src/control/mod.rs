//! Control agents module.
use crate::Shared;
use rand::Rng;

pub trait Controller<S, A> {
    /// Sample the target policy for a given state `s`.
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> A;

    /// Sample the behaviour policy for a given state `s`.
    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> A;
}

impl<S, A, T: Controller<S, A>> Controller<S, A> for Shared<T> {
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> A { self.borrow().sample_target(rng, s) }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> A {
        self.borrow().sample_behaviour(rng, s)
    }
}

pub mod ac;
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
