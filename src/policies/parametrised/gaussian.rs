use core::{Parameter, Handler};
use domains::Transition;
use fa::{Projector, Projection, Approximator, SimpleLFA};
use geometry::{Vector, Matrix};
use rand::{
    thread_rng, ThreadRng,
    distributions::{
        IndependentSample,
        normal::Normal as NormalDist,
    },
};
use std::iter::repeat;
use super::{Policy, DifferentiablePolicy, pdfs::normal_pdf};

pub struct Gaussian1d<S, M: Projector<S>> {
    pub fa_mean: SimpleLFA<S, M>,
    pub std: Parameter,

    rng: ThreadRng,
}

impl<S, M: Projector<S>> Gaussian1d<S, M> {
    pub fn new<T: Into<Parameter>>(fa_mean: SimpleLFA<S, M>, std: T) -> Self {
        Gaussian1d {
            fa_mean: fa_mean,
            std: std.into(),

            rng: thread_rng(),
        }
    }
}

impl<S, M: Projector<S>> Gaussian1d<S, M> {
    pub fn mean(&self, phi: &Projection) -> f64 {
        self.fa_mean.approximator.evaluate(phi).unwrap()
    }
}

impl<S, M: Projector<S>> Handler<Transition<S, f64>> for Gaussian1d<S, M> {
    fn handle_terminal(&mut self, _: &Transition<S, f64>) {
        self.std.step();
    }
}

impl<S, M: Projector<S>> Policy<S, f64> for Gaussian1d<S, M> {
    fn sample(&mut self, input: &S) -> f64 {
        let phi = self.fa_mean.projector.project(input);

        NormalDist::new(self.mean(&phi), self.std.value()).ind_sample(&mut self.rng)
    }

    fn probability(&mut self, input: &S, a: f64) -> f64 {
        let phi = self.fa_mean.projector.project(input);

        normal_pdf(self.mean(&phi), self.std.value(), a)
    }
}

impl<S, M: Projector<S>> DifferentiablePolicy<S, f64> for Gaussian1d<S, M> {
    fn grad_log(&self, input: &S, a: f64) -> Matrix<f64> {
        let phi = self.fa_mean.projector.project(input);
        let mean = self.mean(&phi);

        let n_features = self.fa_mean.projector.dim();
        let phi = phi.expanded(n_features);

        let c = (a - mean) / (self.std * self.std);
        let grad_log_mean = phi.into_iter().map(move |x| c*x);
        let grad_log_std = repeat(0.0).take(n_features);

        Vector::from_iter(grad_log_mean.chain(grad_log_std)).into_shape((n_features, 2)).unwrap()
    }
}
