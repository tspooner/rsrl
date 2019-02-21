use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, VFunction, Parameterised, Projection, Projector, ScalarLFA};
use ndarray::Axis;
use crate::policies::{DifferentiablePolicy, ParameterisedPolicy, Policy};
use rand::{
    distributions::{Distribution, Normal as NormalDist},
    rngs::ThreadRng,
    thread_rng,
};
use std::ops::AddAssign;
use super::pdfs::normal_pdf;

pub struct Mean<F> {
    pub fa: F,
}

impl<F> Mean<F> {
    fn evaluate<S>(&self, input: &S) -> f64
        where F: VFunction<S>,
    {
        self.fa.evaluate(input).unwrap()
    }
}

impl<M> Mean<ScalarLFA<M>> {
    fn grad_log<S>(&self, input: &S, a: f64, std: f64) -> Vector<f64>
        where M: Projector<S>,
    {
        let phi = self.fa.projector.project(input);
        let mean = self.fa.approximator.evaluate(&phi).unwrap();
        let phi = phi.expanded(self.fa.projector.dim());

        let prob = normal_pdf(mean, std, a);

        prob * (a - mean) * phi
    }
}

pub struct Gaussian1d<F> {
    pub mean: Mean<F>,
    pub std: Parameter,

    rng: ThreadRng,
}

impl<F> Gaussian1d<F> {
    pub fn new<T: Into<Parameter>>(fa_mean: F, std: T) -> Self {
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

impl<S, M: Projector<S>> DifferentiablePolicy<S> for Gaussian1d<ScalarLFA<M>> {
    fn grad_log(&self, input: &S, a: f64) -> Matrix<f64> {
        self.mean.grad_log(input, a, self.std()).insert_axis(Axis(1))
    }
}

impl<F: Parameterised> Parameterised for Gaussian1d<F> {
    fn weights(&self) -> Matrix<f64> {
        self.mean.fa.weights()
    }
}

impl<S, M: Projector<S>> ParameterisedPolicy<S> for Gaussian1d<ScalarLFA<M>> {
    fn update(&mut self, input: &S, a: f64, error: f64) {
        let grad_log = self.grad_log(input, a);

        self.mean
            .fa
            .approximator
            .weights
            .scaled_add(error, &grad_log.column(0));
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.mean
            .fa
            .approximator
            .weights
            .add_assign(&errors.column(0))
    }
}
