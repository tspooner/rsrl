use crate::core::*;
use crate::fa::Parameterised;
use crate::geometry::{Space, MatrixView, MatrixViewMut};
use crate::policies::{FinitePolicy, DifferentiablePolicy, ParameterisedPolicy, Policy};
use rand::{
    distributions::{Distribution, Normal},
    rngs::ThreadRng,
    Rng,
    thread_rng,
};
use ndarray::Axis;
use std::ops::Add;

/// Independent Policy Pair (IPP).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IPP<P1, P2>(pub P1, pub P2);

impl<P1, P2> IPP<P1, P2> {
    pub fn new(p1: P1, p2: P2) -> Self { IPP(p1, p2) }
}

impl<P1: Algorithm, P2: Algorithm> Algorithm for IPP<P1, P2> {
    fn handle_terminal(&mut self) {
        self.0.handle_terminal();
        self.1.handle_terminal();
    }
}

impl<S, P1, P2> Policy<S> for IPP<P1, P2>
where
    P1: Policy<S>,
    P2: Policy<S>,
{
    type Action = (P1::Action, P2::Action);

    fn sample(&self, rng: &mut impl Rng, s: &S) -> (P1::Action, P2::Action) {
        (self.0.sample(rng, s), self.1.sample(rng, s))
    }

    fn mpa(&self, s: &S) -> (P1::Action, P2::Action) {
        (self.0.mpa(s), self.1.mpa(s))
    }

    fn probability(&self, s: &S, a: &(P1::Action, P2:: Action)) -> f64 {
        self.0.probability(s, &a.0) * self.1.probability(s, &a.1)
    }
}

impl<S, P1, P2> DifferentiablePolicy<S> for IPP<P1, P2>
where
    P1: DifferentiablePolicy<S>,
    P2: DifferentiablePolicy<S>,
{
    fn grad_log(&self, input: &S, a: &Self::Action) -> Matrix<f64> {
        stack![Axis(0), self.0.grad_log(input, &a.0), self.1.grad_log(input, &a.1)]
    }
}

impl<P1: Parameterised, P2: Parameterised> Parameterised for IPP<P1, P2> {
    fn weights(&self) -> Matrix<f64> {
        stack![Axis(1), self.0.weights(), self.1.weights()]
    }

    fn weights_view(&self) -> MatrixView<f64> {
        unimplemented!()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }

    fn weights_dim(&self) -> (usize, usize) {
        let d0 = self.0.weights_dim();
        let d1 = self.0.weights_dim();

        (d0.0 + d1.0, d0.1 + d1.1)
    }
}

impl<S, P1, P2> ParameterisedPolicy<S> for IPP<P1, P2>
where
    P1: ParameterisedPolicy<S>,
    P2: ParameterisedPolicy<S>,
{
    fn update(&mut self, input: &S, a: &Self::Action, error: f64) {
        self.0.update(input, &a.0, error);
        self.1.update(input, &a.1, error);
    }
}
