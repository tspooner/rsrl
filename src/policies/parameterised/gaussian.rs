use crate::{
    core::*,
    domains::Transition,
    fa::{Approximator, VFunction, Parameterised, Projector},
    geometry::{MatrixView, MatrixViewMut},
    policies::{DifferentiablePolicy, ParameterisedPolicy, Policy},
};
use ndarray::Axis;
use rand::{
    distributions::{Distribution, Normal as NormalDist},
    rngs::ThreadRng,
    thread_rng,
};
use std::ops::AddAssign;
use super::pdfs::normal_pdf;

pub struct Mean<F> {
    pub fa: Shared<F>,
}

impl<F> Mean<F> {
    fn evaluate<S>(&self, input: &S) -> f64
        where F: VFunction<S>,
    {
        self.fa.evaluate(&self.fa.to_features(input)).unwrap()
    }
}

impl<F> Mean<F> {
    fn grad_log<S>(&self, input: &S, a: f64, std: f64) -> Vector<f64>
        where F: VFunction<S>,
    {
        let phi = self.fa.to_features(input);
        let mean = self.fa.evaluate(&phi).unwrap();
        let phi = phi.expanded(self.fa.n_features());

        (a - mean) / std / std * phi
    }
}

pub struct Gaussian1d<F> {
    pub mean: Mean<F>,
    pub std: Parameter,

    rng: ThreadRng,
}

impl<F> Gaussian1d<F> {
    pub fn new<T: Into<Parameter>>(fa_mean: Shared<F>, std: T) -> Self {
        Gaussian1d {
            mean: Mean { fa: fa_mean, },
            std: std.into(),

            rng: thread_rng(),
        }
    }

    #[inline]
    pub fn mean<S>(&self, input: &S) -> f64
        where F: VFunction<S>,
    {
        self.mean.evaluate(input)
    }

    #[inline]
    pub fn std(&self) -> f64 {
        self.std.value()
    }

    #[inline]
    pub fn var(&self) -> f64 {
        let std = self.std();

        std * std
    }
}

impl<F> Algorithm for Gaussian1d<F> {
    fn handle_terminal(&mut self) {
        self.std = self.std.step();
    }
}

impl<S, F: VFunction<S>> Policy<S> for Gaussian1d<F> {
    type Action = f64;

    fn sample(&mut self, input: &S) -> f64 {
        let mean = self.mean(input);

        NormalDist::new(mean, self.std()).sample(&mut self.rng)
    }

    fn mpa(&mut self, input: &S) -> f64 {
        self.mean(input)
    }

    fn probability(&mut self, input: &S, a: f64) -> f64 {
        let mean = self.mean(input);

        normal_pdf(mean, self.std(), a)
    }
}

impl<S, F: VFunction<S>> DifferentiablePolicy<S> for Gaussian1d<F> {
    fn grad_log(&self, input: &S, a: f64) -> Matrix<f64> {
        self.mean.grad_log(input, a, self.std()).insert_axis(Axis(1))
    }
}

impl<F: Parameterised> Parameterised for Gaussian1d<F> {
    fn weights(&self) -> Matrix<f64> {
        self.mean.fa.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.mean.fa.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }
}

impl<S, F: VFunction<S> + Parameterised> ParameterisedPolicy<S> for Gaussian1d<F> {
    fn update(&mut self, input: &S, a: f64, error: f64) {
        let grad_log = self.grad_log(input, a);

        let mut fa = self.mean.fa.borrow_mut();
        let mut weights = fa.weights_view_mut();

        weights.scaled_add(error, &grad_log);
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        let mut fa = self.mean.fa.borrow_mut();
        let mut weights = fa.weights_view_mut();

        weights.add_assign(&errors);
    }
}
