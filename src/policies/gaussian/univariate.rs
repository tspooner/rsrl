use crate::{
    core::{Algorithm, Parameter},
    fa::{Embedded, Features, Parameterised, VFunction},
    geometry::{Matrix, MatrixView, MatrixViewMut},
    policies::{DifferentiablePolicy, ParameterisedPolicy, Policy},
};
use ndarray::Axis;
use rand::{thread_rng, rngs::{ThreadRng}};
use rstat::{
    Distribution, ContinuousDistribution,
    univariate::continuous::Normal,
};
use super::Mean;
use std::ops::AddAssign;

pub struct UnivariateGaussian<F> {
    mean: Mean<F>,
    std: Parameter,

    rng: ThreadRng,
}

impl<F> UnivariateGaussian<F> {
    pub fn new<T: Into<Parameter>>(fa_mean: F, std: T) -> Self {
        UnivariateGaussian {
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

    #[inline]
    fn dist<S>(&self, input: &S) -> Normal
        where F: VFunction<S>,
    {
        Normal::new(self.mean(input), self.std())
    }
}

impl<F> Algorithm for UnivariateGaussian<F> {
    fn handle_terminal(&mut self) {
        self.std = self.std.step();
    }
}

impl<S, F: Embedded<S>> Embedded<S> for UnivariateGaussian<F> {
    fn n_features(&self) -> usize {
        self.mean.fa.n_features()
    }

    fn to_features(&self, s: &S) -> Features {
        self.mean.fa.to_features(s)
    }
}

impl<S, F: VFunction<S>> Policy<S> for UnivariateGaussian<F> {
    type Action = f64;

    fn sample(&mut self, input: &S) -> f64 {
        self.dist(input).sample(&mut self.rng)
    }

    fn mpa(&mut self, input: &S) -> f64 {
        self.mean(input)
    }

    fn probability(&mut self, input: &S, a: f64) -> f64 {
        f64::from(self.dist(input).pdf(a))
    }
}

impl<S, F: VFunction<S>> DifferentiablePolicy<S> for UnivariateGaussian<F> {
    fn grad_log(&self, input: &S, a: f64) -> Matrix<f64> {
        self.mean.grad_log(input, a, self.std()).insert_axis(Axis(1))
    }
}

impl<F: Parameterised> Parameterised for UnivariateGaussian<F> {
    fn weights(&self) -> Matrix<f64> {
        self.mean.fa.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.mean.fa.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.mean.fa.weights_view_mut()
    }
}

impl<S, F: VFunction<S> + Parameterised> ParameterisedPolicy<S> for UnivariateGaussian<F> {
    fn update(&mut self, input: &S, a: f64, error: f64) {
        let grad_log = self.grad_log(input, a);

        self.mean
            .fa
            .weights_view_mut()
            .scaled_add(error, &grad_log);
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.mean
            .fa
            .weights_view_mut()
            .add_assign(&errors)
    }
}
