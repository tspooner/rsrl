use crate::{
    core::*,
    fa::{Approximator, VFunction, Parameterised, Embedded, Features},
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
    pub fa: F,
}

impl<F> Dirac<F> {
    pub fn new(fa: F) -> Self {
        Dirac { fa, }
    }
}

impl<S, F: Embedded<S>> Embedded<S> for Dirac<F> {
    fn n_features(&self) -> usize {
        self.fa.n_features()
    }

    fn to_features(&self, s: &S) -> Features {
        self.fa.to_features(s)
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
        self.fa.to_features(input).expanded(self.fa.n_features()).insert_axis(Axis(1))
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
        self.fa.weights_view_mut()
    }
}

impl<S, F: VFunction<S> + Parameterised> ParameterisedPolicy<S> for Dirac<F> {
    fn update(&mut self, input: &S, _: f64, error: f64) {
        self.fa.update(&self.fa.to_features(input), error).ok();
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.fa.weights_view_mut().add_assign(&errors);
    }
}
