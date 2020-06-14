use crate::{spaces::Space, Function};
use rstat::{
    builder::{BuildNormal, Builder},
    Distribution,
};
use std::fmt::Debug;

type BuilderDist<M, S> = <Builder as BuildNormal<M, S>>::Normal;
type BuilderSupport<M, S> = <BuilderDist<M, S> as Distribution>::Support;

const MIN_TOL: f64 = 0.01;

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Gaussian<M, S> {
    pub mean: M,
    pub stddev: S,
}

impl<M, S> Gaussian<M, S> {
    pub fn new(mean: M, stddev: S) -> Self { Gaussian { mean, stddev } }
}

impl<M, S> Gaussian<M, S> {
    #[inline]
    pub fn compute_mean<X>(&self, x: X) -> M::Output
    where M: Function<(X,)> {
        self.mean.evaluate((x,))
    }

    #[inline]
    pub fn compute_stddev<X>(&self, x: X) -> S::Output
    where
        S: Function<(X,)>,
        S::Output: std::ops::Add<f64, Output = S::Output>,
    {
        self.stddev.evaluate((x,)) + MIN_TOL
    }

    #[inline]
    fn dist<'x, X>(&self, x: &'x X) -> <Builder as BuildNormal<M::Output, S::Output>>::Normal
    where
        M: Function<(&'x X,)>,
        S: Function<(&'x X,)>,

        M::Output: Clone,
        S::Output: std::ops::Add<f64, Output = S::Output>,

        Builder: BuildNormal<M::Output, S::Output>,
        BuilderSupport<M::Output, S::Output>: Space<Value = M::Output>,
    {
        Builder::build_unchecked(self.compute_mean(x), self.compute_stddev(x))
    }
}

mod fixed_var;
mod general;
