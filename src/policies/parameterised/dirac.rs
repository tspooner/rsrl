use crate::{
    core::*,
    fa::{Approximator, VFunction, Parameterised, Projector},
    geometry::{MatrixView, MatrixViewMut},
    policies::{
        DifferentiablePolicy,
        ParameterisedPolicy,
        Policy
    },
};
use ndarray::Axis;
use std::ops::AddAssign;

pub struct Dirac<F> {
    pub fa: Shared<F>,
}

impl<F> Dirac<F> {
    pub fn new(fa: Shared<F>) -> Self {
        Dirac { fa, }
    }
}

impl<F> Algorithm for Dirac<F> {}

impl<S, F: VFunction<S>> Policy<S> for Dirac<F> {
    type Action = f64;

    fn mpa(&mut self, s: &S) -> f64 {
        self.fa.evaluate(&self.fa.to_features(s)).unwrap()
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

impl<S, F: VFunction<S>> DifferentiablePolicy<S> for Dirac<F> {
    fn grad_log(&self, input: &S, _: f64) -> Matrix<f64> {
        self.fa.to_features(input).expanded(self.fa.n_features()).insert_axis(Axis(0))
    }
}

impl<F: Parameterised> Parameterised for Dirac<F> {
    fn weights(&self) -> Matrix<f64> {
        self.fa.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.fa.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }
}

impl<S, F: VFunction<S> + Parameterised> ParameterisedPolicy<S> for Dirac<F> {
    fn update(&mut self, input: &S, _: f64, error: f64) {
        self.fa.borrow_mut().update(&self.fa.to_features(input), error).ok();
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        let mut fa = self.fa.borrow_mut();

        fa.weights_view_mut().add_assign(&errors);
    }
}
