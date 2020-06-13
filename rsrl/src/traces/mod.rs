//! Eligibility trace types.
use crate::params::{Buffer, BufferMut};
use ndarray::{ArrayBase, Array, Dimension, IntoDimension, DataMut};

pub struct Trace<B: BufferMut, R: UpdateRule<B>> {
    pub buffer: B,
    pub update_rule: R,
}

impl<B, R> Trace<B, R>
where
    B: BufferMut,
    R: UpdateRule<B>,
{
    pub fn new(buffer: B, update_rule: R) -> Self {
        Trace { buffer, update_rule, }
    }

    pub fn zeros<D>(dim: D, update_rule: R) -> Self
    where D: IntoDimension<Dim = B::Dim>,
    {
        Trace::new(B::zeros(dim), update_rule)
    }
}

impl<B: BufferMut> Trace<B, Accumulate> {
    pub fn accumulating<D>(dim: D, gamma: f64, lambda: f64) -> Self
    where D: IntoDimension<Dim = B::Dim>,
    {
        Trace::zeros(dim, Accumulate { gamma, lambda, })
    }
}

impl<B: BufferMut> Trace<B, Saturate> {
    pub fn replacing<D>(dim: D, gamma: f64, lambda: f64) -> Self
    where D: IntoDimension<Dim = B::Dim>,
    {
        Trace::zeros(dim, Saturate { gamma, lambda, })
    }
}

impl<B: BufferMut> Trace<B, Dutch> {
    pub fn dutch<D>(dim: D, alpha: f64, gamma: f64, lambda: f64) -> Self
    where D: IntoDimension<Dim = B::Dim>,
    {
        Trace::zeros(dim, Dutch { alpha, gamma, lambda, })
    }
}

impl<B, R> Trace<B, R>
where
    B: BufferMut,
    R: UpdateRule<B>,
{
    pub fn update(&mut self, buffer: &B) {
        self.update_rule.update_trace(&mut self.buffer, buffer)
    }

    pub fn reset(&mut self) { self.buffer.reset() }
}

impl<B: BufferMut, R: UpdateRule<B>> Buffer for Trace<B, R> {
    type Dim = B::Dim;

    fn dim(&self) -> <B::Dim as Dimension>::Pattern { self.buffer.dim() }

    fn n_dim(&self) -> usize { self.buffer.n_dim() }

    fn raw_dim(&self) -> B::Dim { self.buffer.raw_dim() }

    fn addto<D: DataMut<Elem = f64>>(&self, weights: &mut ArrayBase<D, B::Dim>) {
        self.buffer.addto(weights)
    }

    fn scaled_addto<D>(&self, alpha: f64, weights: &mut ArrayBase<D, B::Dim>)
    where D: DataMut<Elem = f64>,
    {
        self.buffer.scaled_addto(alpha, weights)
    }

    fn to_dense(&self) -> Array<f64, B::Dim> { self.buffer.to_dense() }

    fn into_dense(self) -> Array<f64, B::Dim> { self.buffer.into_dense() }
}

pub trait UpdateRule<B: BufferMut> {
    fn update_trace(&self, trace: &mut B, buffer: &B);
}

pub struct Accumulate {
    pub gamma: f64,
    pub lambda: f64,
}

impl<B: BufferMut> UpdateRule<B> for Accumulate {
    fn update_trace(&self, trace: &mut B, buffer: &B) {
        let rate = self.gamma * self.lambda;

        trace.merge_inplace(buffer, |x, y| rate * x + y)
    }
}

pub struct Saturate {
    pub gamma: f64,
    pub lambda: f64,
}

impl<B: BufferMut> UpdateRule<B> for Saturate {
    fn update_trace(&self, trace: &mut B, buffer: &B) {
        let rate = self.gamma * self.lambda;

        trace
            .merge_inplace(buffer, |x, y| f64::max(-1.0, f64::min(1.0, rate * x + y)));
    }
}

pub struct Dutch {
    pub alpha: f64,
    pub gamma: f64,
    pub lambda: f64,
}

impl<B: BufferMut> UpdateRule<B> for Dutch {
    fn update_trace(&self, trace: &mut B, buffer: &B) {
        let rate = self.gamma * self.lambda * (1.0 - self.alpha);

        trace.merge_inplace(buffer, |x, y| rate * x + y)
    }
}
