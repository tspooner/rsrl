use core::*;
use domains::Transition;
use fa::{Approximator, Parameterised, Projection, Projector, SimpleLFA};
use ndarray::Axis;
use policies::{DifferentiablePolicy, ParameterisedPolicy, Policy};
use rand::{
    distributions::{Distribution, Normal as NormalDist},
    rngs::ThreadRng,
    thread_rng,
};
use std::ops::AddAssign;
use super::pdfs::normal_pdf;

struct Mean<S, M: Projector<S>> {
    pub fa: SimpleLFA<S, M>,
}

impl<S, M: Projector<S>> Mean<S, M> {
    fn evaluate(&self, input: &S) -> f64 {
        self.fa.evaluate(input).unwrap()
    }

    fn grad_log(&self, input: &S, a: f64) -> Vector<f64> {
        let phi = self.fa.projector.project(input);
        let mean = self.fa.approximator.evaluate(&phi).unwrap();
        let phi = phi.expanded(self.fa.projector.dim());

        (a - mean) * phi
    }
}

pub struct Gaussian1d<S, M: Projector<S>> {
    mean: Mean<S, M>,
    std: Parameter,

    rng: ThreadRng,
}

impl<S, M: Projector<S>> Gaussian1d<S, M> {
    pub fn new<T: Into<Parameter>>(fa_mean: SimpleLFA<S, M>, std: T) -> Self {
        Gaussian1d {
            mean: Mean { fa: fa_mean, },
            std: std.into(),

            rng: thread_rng(),
        }
    }

    #[inline]
    pub fn mean(&self, input: &S) -> f64 {
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

impl<S, M: Projector<S>> Algorithm for Gaussian1d<S, M> {
    fn handle_terminal(&mut self) { self.std = self.std.step(); }
}

impl<S, M: Projector<S>> Policy<S> for Gaussian1d<S, M> {
    type Action = f64;

    fn sample(&mut self, input: &S) -> f64 {
        let mean = self.mean(input);

        NormalDist::new(mean, self.std()).sample(&mut self.rng)
    }

    fn probability(&mut self, input: &S, a: f64) -> f64 {
        let mean = self.mean(input);

        normal_pdf(mean, self.std(), a)
    }
}

impl<S, M: Projector<S>> DifferentiablePolicy<S> for Gaussian1d<S, M> {
    fn grad_log(&self, input: &S, a: f64) -> Matrix<f64> {
        self.mean.grad_log(input, a).insert_axis(Axis(1))
    }
}

impl<S, M: Projector<S>> Parameterised for Gaussian1d<S, M> {
    fn weights(&self) -> Matrix<f64> {
        self.mean.fa.weights()
    }
}

impl<S, M: Projector<S>> ParameterisedPolicy<S> for Gaussian1d<S, M> {
    fn update(&mut self, input: &S, a: f64, error: f64) {
        let pi = self.probability(input, a);
        let grad_log = self.grad_log(input, a);

        self.mean
            .fa
            .approximator
            .weights
            .scaled_add(pi * error, &grad_log.column(0));
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.mean
            .fa
            .approximator
            .weights
            .add_assign(&errors.column(0))
    }
}
