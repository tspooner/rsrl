use crate::{
    core::{Algorithm, Parameter},
    fa::{Approximator, Embedded, Features, Parameterised, VFunction},
    geometry::{Space, Matrix, MatrixView, MatrixViewMut, Vector, continuous::Interval},
    policies::{DifferentiablePolicy, ParameterisedPolicy, Policy},
};
use ndarray::Axis;
use rand::{thread_rng, rngs::{ThreadRng}};
use rstat::{
    Distribution, ContinuousDistribution,
    univariate::continuous::Normal,
};
use std::{fmt::Debug, ops::AddAssign};

pub mod mean;
use self::mean::Mean;

pub mod stddev;
use self::stddev::StdDev;

import_all!(dbuilder);

pub struct Gaussian<M, S> {
    mean: M,
    stddev: S,

    rng: ThreadRng,
}

impl<M, S> Gaussian<M, S> {
    pub fn new(mean: M, stddev: S) -> Self {
        Gaussian {
            mean, stddev,

            rng: thread_rng(),
        }
    }
}

impl<M, S> Gaussian<M, S> {
    #[inline]
    pub fn mean<I>(&self, input: &I) -> M::Output
    where
        M: Mean<I, <S as Approximator>::Output>,
        S: StdDev<I, <M as Approximator>::Output>,
    {
        self.mean.mean(input)
    }

    #[inline]
    pub fn stddev<I>(&self, input: &I) -> S::Output
    where
        M: Mean<I, <S as Approximator>::Output>,
        S: StdDev<I, <M as Approximator>::Output>,
    {
        self.stddev.stddev(input)
    }
}

impl<M, S> Algorithm for Gaussian<M, S> {}

impl<I, M: Embedded<I>, S: Embedded<I>> Embedded<I> for Gaussian<M, S> {
    fn n_features(&self) -> usize {
        self.mean.n_features() + self.stddev.n_features()
    }

    fn to_features(&self, input: &I) -> Features {
        let phi_mean = self.mean.to_features(input).expanded(self.mean.n_features());
        let phi_stddev = self.stddev.to_features(input).expanded(self.stddev.n_features());

        Features::Dense(stack![Axis(0), phi_mean, phi_stddev])
    }
}

impl<I, M, S> Policy<I> for Gaussian<M, S>
where
    M: Mean<I, <S as Approximator>::Output>,
    M::Output: Clone + Debug,
    S: StdDev<I, <M as Approximator>::Output>,
    S::Output: Clone + Debug,
    GB: DistBuilder<M::Output, S::Output>,
    GBSupport<M::Output, S::Output>: Space<Value = M::Output>
{
    type Action = M::Output;

    fn sample(&mut self, input: &I) -> Self::Action {
        GB::build(self.mean(input), self.stddev(input)).sample(&mut self.rng)
    }

    fn mpa(&mut self, input: &I) -> Self::Action {
        self.mean(input)
    }

    fn probability(&mut self, input: &I, a: Self::Action) -> f64 {
        GB::build(self.mean(input), self.stddev(input)).pdf(a)
    }
}

impl<I, M, S> DifferentiablePolicy<I> for Gaussian<M, S>
where
    M: Mean<I, <S as Approximator>::Output>,
    M::Output: Clone + Debug,
    S: StdDev<I, <M as Approximator>::Output>,
    S::Output: Clone + Debug,
    GB: DistBuilder<M::Output, S::Output>,
    GBSupport<M::Output, S::Output>: Space<Value = M::Output>
{
    fn grad_log(&self, input: &I, a: Self::Action) -> Matrix<f64> {
        let mean = self.mean(input);
        let stddev = self.stddev(input);

        stack![
            Axis(1),
            self.mean.grad_log(input, &a, stddev).insert_axis(Axis(1)),
            self.stddev.grad_log(input, &a, mean).insert_axis(Axis(1))
        ]
    }
}

impl<M, S> Parameterised for Gaussian<M, S>
where
    M: Parameterised,
    S: Parameterised,
{
    fn weights(&self) -> Matrix<f64> {
        stack![
            Axis(1),
            self.mean.weights(),
            self.stddev.weights()
        ]
    }

    fn weights_view(&self) -> MatrixView<f64> {
        unimplemented!()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }
}

impl<I, M, S> ParameterisedPolicy<I> for Gaussian<M, S>
where
    M: Mean<I, <S as Approximator>::Output> + Parameterised,
    M::Output: Clone + Debug,
    S: StdDev<I, <M as Approximator>::Output> + Parameterised,
    S::Output: Clone + Debug,
    GB: DistBuilder<M::Output, S::Output>,
    GBSupport<M::Output, S::Output>: Space<Value = M::Output>
{
    fn update(&mut self, input: &I, a: Self::Action, error: f64) {
        let mean = self.mean(input);
        let stddev = self.stddev(input);

        self.mean.update_mean(input, &a, stddev, error);
        self.stddev.update_stddev(input, &a, mean, error);
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.mean.weights_view_mut().add_assign(&errors.column(0).insert_axis(Axis(1)));
        self.stddev.weights_view_mut().add_assign(&errors.column(1).insert_axis(Axis(1)));
    }
}
