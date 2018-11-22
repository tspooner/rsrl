use super::pdfs::normal_pdf;
use core::Parameter;
use domains::Transition;
use fa::{Approximator, Parameterised, Projection, Projector, SimpleLFA};
use geometry::Matrix;
use policies::{DifferentiablePolicy, ParameterisedPolicy, Policy};
use rand::{
    distributions::{Normal as NormalDist, Distribution},
    rngs::ThreadRng,
    thread_rng,
};
use std::ops::AddAssign;

pub struct Gaussian1d<S, M: Projector<S>> {
    pub fa_mean: SimpleLFA<S, M>,
    pub std: Parameter,

    rng: ThreadRng,
}

impl<S, M: Projector<S>> Gaussian1d<S, M> {
    pub fn new<T: Into<Parameter>>(fa_mean: SimpleLFA<S, M>, std: T) -> Self {
        Gaussian1d {
            fa_mean,
            std: std.into(),

            rng: thread_rng(),
        }
    }
}

impl<S, M: Projector<S>> Gaussian1d<S, M> {
    pub fn mean(&self, phi: &Projection) -> f64 { self.fa_mean.approximator.evaluate(phi).unwrap() }
}

impl<S, M: Projector<S>> Policy<S> for Gaussian1d<S, M> {
    type Action = f64;

    fn sample(&mut self, input: &S) -> f64 {
        let phi = self.fa_mean.projector.project(input);
        let mean = self.mean(&phi);

        NormalDist::new(mean, self.std.value()).sample(&mut self.rng)
    }

    fn probability(&mut self, input: &S, a: f64) -> f64 {
        let phi = self.fa_mean.projector.project(input);

        normal_pdf(self.mean(&phi), self.std.value(), a)
    }

    fn handle_terminal(&mut self, _: &Transition<S, f64>) { self.std.step(); }
}

impl<S, M: Projector<S>> DifferentiablePolicy<S> for Gaussian1d<S, M> {
    fn grad_log(&self, input: &S, a: f64) -> Matrix<f64> {
        // let phi = self.fa_mean.projector.project(input);
        // let mean = self.mean(&phi);

        // let n_rows = self.fa_mean.projector.dim();
        // let phi = phi.expanded(n_rows);

        // let c = (a - mean) / (self.std * self.std);
        // let grad_log_mean = phi.into_iter().map(move |x| c*x);
        // let grad_log_std = repeat(0.0).take(n_rows);

        // Vector::from_iter(grad_log_mean.chain(grad_log_std)).into_shape((n_rows,
        // 2)).unwrap()

        let phi = self.fa_mean.projector.project(input);
        let mean = self.mean(&phi);
        let std = self.std.value();

        let n_rows = self.fa_mean.projector.dim();
        let phi = phi.expanded(n_rows);

        ((a - mean) / (std * std) * phi)
            .into_shape((n_rows, 1))
            .unwrap()
    }
}

impl<S, M: Projector<S>> Parameterised for Gaussian1d<S, M> {
    fn weights(&self) -> Matrix<f64> {
        // let mean_col = self.fa_mean.approximator.weights.clone();
        // let n_rows = mean_col.len();

        // let mean_col = mean_col.to_vec().into_iter();
        // let std_col = repeat(self.std.value()).take(n_rows);

        // Vector::from_iter(mean_col.chain(std_col)).into_shape((n_rows, 2)).unwrap()

        let mean_col = self.fa_mean.approximator.weights.clone();
        let n_rows = mean_col.len();

        mean_col.into_shape((n_rows, 1)).unwrap()
    }
}

impl<S, M: Projector<S>> ParameterisedPolicy<S> for Gaussian1d<S, M> {
    fn update(&mut self, input: &S, a: f64, error: f64) {
        let pi = self.probability(input, a);
        let grad_log = self.grad_log(input, a);

        self.fa_mean
            .approximator
            .weights
            .scaled_add(pi * error, &grad_log.column(0));
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.fa_mean
            .approximator
            .weights
            .add_assign(&errors.column(0))
    }
}
