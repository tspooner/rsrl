use crate::{
    core::*,
    fa::{Approximator, ScalarLFA, Parameterised, Projector},
    policies::{
        DifferentiablePolicy,
        ParameterisedPolicy,
        Policy
    },
};
use ndarray::Axis;
use std::ops::AddAssign;

pub struct Dirac<F> {
    pub fa: F,
}

impl<F> Dirac<F> {
    pub fn new(fa: F) -> Self {
        Dirac {
            fa,
        }
    }
}

impl<F> Algorithm for Dirac<F> {}

impl<S, M: Projector<S>> Policy<S> for Dirac<ScalarLFA<M>> {
    type Action = f64;

    fn mpa(&mut self, s: &S) -> f64 {
        self.fa.evaluate(s).unwrap()
    }

    fn probability(&mut self, input: &S, a: f64) -> f64 {
        let mpa = self.mpa(input);

        if (a - mpa).abs() < 1e-7 {
            1.0
        } else {
            0.0
        }
    }
}

impl<S, M: Projector<S>> DifferentiablePolicy<S> for Dirac<ScalarLFA<M>> {
    fn grad_log(&self, input: &S, _: f64) -> Matrix<f64> {
        self.fa.projector.project_expanded(input).insert_axis(Axis(0))
    }
}

impl<F: Parameterised> Parameterised for Dirac<F> {
    fn weights(&self) -> Matrix<f64> { self.fa.weights() }
}

impl<S, M: Projector<S>> ParameterisedPolicy<S> for Dirac<ScalarLFA<M>> {
    fn update(&mut self, input: &S, _: f64, error: f64) {
        self.fa.update(input, error).ok();
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.fa.approximator.weights.add_assign(&errors)
    }
}
