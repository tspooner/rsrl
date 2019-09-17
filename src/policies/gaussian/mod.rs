use crate::{
    fa::{Parameterised, StateFunction, Weights, WeightsView, WeightsViewMut},
    policies::{DifferentiablePolicy, Policy},
    spaces::Space,
    Algorithm,
};
use ndarray::{Array2, ArrayView2, Axis};
use rand::Rng;
use rstat::{ContinuousDistribution, Distribution};
use std::fmt::Debug;

pub mod mean;
use self::mean::Mean;

pub mod stddev;
use self::stddev::StdDev;

import_all!(dbuilder);

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Gaussian<M, S> {
    pub mean: M,
    pub stddev: S,

    pub alpha: f64,
}

impl<M, S> Gaussian<M, S> {
    pub fn new(mean: M, stddev: S) -> Self {
        Gaussian {
            mean,
            stddev,
            alpha: 1.0,
        }
    }
}

impl<M, S> Gaussian<M, S> {
    #[inline]
    pub fn compute_mean<I>(&self, input: &I) -> M::Output
    where
        M: Mean<I, <S as StateFunction<I>>::Output>,
        S: StdDev<I, <M as StateFunction<I>>::Output>,
    {
        self.mean.mean(input)
    }

    #[inline]
    pub fn compute_stddev<I>(&self, input: &I) -> S::Output
    where
        M: Mean<I, <S as StateFunction<I>>::Output>,
        S: StdDev<I, <M as StateFunction<I>>::Output>,
    {
        self.stddev.stddev(input)
    }
}

impl<M, S> Algorithm for Gaussian<M, S> {}

impl<I, M, S> Policy<I> for Gaussian<M, S>
where
    M: Mean<I, <S as StateFunction<I>>::Output>,
    M::Output: Clone + Debug,
    S: StdDev<I, <M as StateFunction<I>>::Output>,
    S::Output: Clone + Debug,
    GB: DistBuilder<M::Output, S::Output>,
    GBSupport<M::Output, S::Output>: Space<Value = M::Output>,
{
    type Action = M::Output;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, input: &I) -> Self::Action {
        GB::build(self.compute_mean(input), self.compute_stddev(input)).sample(rng)
    }

    fn mpa(&self, input: &I) -> Self::Action { self.compute_mean(input) }

    fn probability(&self, input: &I, a: &Self::Action) -> f64
    where Self::Action: Clone {
        GB::build(self.compute_mean(input), self.compute_stddev(input)).pdf(a.clone())
    }
}

impl<I, M, S> DifferentiablePolicy<I> for Gaussian<M, S>
where
    M: Mean<I, <S as StateFunction<I>>::Output>,
    M::Output: Clone + Debug,
    S: StdDev<I, <M as StateFunction<I>>::Output>,
    S::Output: Clone + Debug,
    GB: DistBuilder<M::Output, S::Output>,
    GBSupport<M::Output, S::Output>: Space<Value = M::Output>,
{
    fn update(&mut self, input: &I, a: &Self::Action, error: f64) {
        let mean = self.compute_mean(input);
        let stddev = self.compute_stddev(input);

        self.mean.update_mean(input, &a, stddev, error);
        self.stddev.update_stddev(input, &a, mean, error);
    }

    fn update_grad(&mut self, grad: &ArrayView2<f64>) {
        let w_mean = self.mean.weights_dim();

        match w_mean {
            [r, c] if r > 0 => {
                let grad_mean = grad.slice(s![0..r, 0..c]);

                self.mean
                    .weights_view_mut()
                    .scaled_add(self.alpha, &grad_mean);
            },
            _ => {},
        }

        match self.stddev.weights_dim() {
            [r, c] if r > 0 => {
                let grad_stddev = grad.slice(s![w_mean[0]..(w_mean[0] + r), 0..c]);

                self.stddev
                    .weights_view_mut()
                    .scaled_add(self.alpha, &grad_stddev);
            },
            _ => {},
        }
    }

    fn update_grad_scaled(&mut self, grad: &ArrayView2<f64>, factor: f64) {
        let w_mean = self.mean.weights_dim();

        match w_mean {
            [r, c] if r > 0 => {
                let grad_mean = grad.slice(s![0..r, 0..c]);

                self.mean
                    .weights_view_mut()
                    .scaled_add(self.alpha * factor, &grad_mean);
            },
            _ => {},
        }

        match self.stddev.weights_dim() {
            [r, c] if r > 0 => {
                let grad_stddev = grad.slice(s![w_mean[0]..(w_mean[0] + r), 0..c]);

                self.stddev
                    .weights_view_mut()
                    .scaled_add(self.alpha * factor, &grad_stddev);
            },
            _ => {},
        }
    }

    fn grad_log(&self, input: &I, a: &Self::Action) -> Array2<f64> {
        let mean = self.compute_mean(input);
        let stddev = self.compute_stddev(input);

        let gl_mean = self.mean.grad_log(input, &a, stddev);
        let gl_stddev = self.stddev.grad_log(input, &a, mean);

        if gl_stddev.len() == 0 {
            gl_mean
        } else {
            stack![Axis(0), gl_mean, gl_stddev]
        }
    }
}

impl<M, S> Parameterised for Gaussian<M, S>
where
    M: Parameterised,
    S: Parameterised,
{
    fn weights(&self) -> Weights {
        let w_mean = self.mean.weights();
        let w_stddev = self.stddev.weights();

        if w_stddev.len() == 0 {
            w_mean
        } else {
            stack![Axis(0), w_mean, w_stddev]
        }
    }

    fn weights_view(&self) -> WeightsView { unimplemented!() }

    fn weights_view_mut(&mut self) -> WeightsViewMut { unimplemented!() }

    fn weights_dim(&self) -> [usize; 2] {
        let [rm, cm] = self.mean.weights_dim();
        let [rs, cs] = self.stddev.weights_dim();

        [rm + rs, cm.max(cs)]
    }
}
