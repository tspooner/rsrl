//! Eligibility trace types.
use crate::params::{Buffer, BufferMut};
use ndarray::{ArrayBase, Array, Dimension, IntoDimension, DataMut};

/// Eligibility trace buffer.
pub struct Trace<B: BufferMut, R: UpdateRule<B>> {
    /// Internal gradient buffer.
    pub buffer: B,

    /// Eligibility update rule.
    pub update_rule: R,
}

impl<B, R> Trace<B, R>
where
    B: BufferMut,
    R: UpdateRule<B>,
{
    /// Construct a new eligibility trace.
    ///
    /// # Arguments
    ///
    /// * `buffer` - A gradient buffer instance.
    /// * `update_rule` - The eligibility update rule.
    ///
    /// # Example
    ///
    /// ```
    /// use rsrl::{params::Vector, traces::{Trace, Accumulate}};
    ///
    /// let mut trace = Trace::new(Vector::zeros(10), Accumulate {
    ///     gamma: 0.95,
    ///     lambda: 0.7,
    /// });
    /// ```
    pub fn new(buffer: B, update_rule: R) -> Self {
        Trace { buffer, update_rule, }
    }

    /// Construct a new eligibility trace with empty gradient buffer.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimensionality of the buffer.
    /// * `update_rule` - The eligibility update rule.
    pub fn zeros<D>(dim: D, update_rule: R) -> Self
    where D: IntoDimension<Dim = B::Dim>,
    {
        Trace::new(B::zeros(dim), update_rule)
    }
}

impl<B: BufferMut> Trace<B, Accumulate> {
    /// Construct a new eligibility trace with empty buffer and accumulation rule.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimensionality of the buffer.
    /// * `gamma` - Discount factor.
    /// * `lambda` - Forgetting rate.
    pub fn accumulating<D>(dim: D, gamma: f64, lambda: f64) -> Self
    where D: IntoDimension<Dim = B::Dim>,
    {
        Trace::zeros(dim, Accumulate { gamma, lambda, })
    }
}

impl<B: BufferMut> Trace<B, Saturate> {
    /// Construct a new eligibility trace with empty buffer and replacement rule.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimensionality of the buffer.
    /// * `gamma` - Discount factor.
    /// * `lambda` - Forgetting rate.
    pub fn replacing<D>(dim: D, gamma: f64, lambda: f64) -> Self
    where D: IntoDimension<Dim = B::Dim>,
    {
        Trace::zeros(dim, Saturate { gamma, lambda, })
    }
}

impl<B: BufferMut> Trace<B, Dutch> {
    /// Construct a new eligibility trace with empty buffer and dutch rule.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimensionality of the buffer.
    /// * `alpha` - Learning rate.
    /// * `gamma` - Discount factor.
    /// * `lambda` - Forgetting rate.
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
    /// Update the trace with a new `buffer` instance.
    ///
    /// # Arguments
    ///
    /// * `buffer` - New gradient buffer instance.
    ///
    /// # Example
    ///
    /// ```
    /// use approx::assert_abs_diff_eq;
    /// use rsrl::{params::Vector, traces::{Trace, Accumulate}};
    ///
    /// let mut trace = Trace::new(Vector::zeros(1), Accumulate {
    ///     gamma: 0.95,
    ///     lambda: 0.7,
    /// });
    ///
    /// trace.update(&Vector::ones(1));
    /// assert_abs_diff_eq!(trace.buffer[0], 1.0);
    ///
    /// trace.update(&Vector::zeros(1));
    /// assert_abs_diff_eq!(trace.buffer[0], 0.665);
    /// ```
    pub fn update(&mut self, buffer: &B) {
        self.update_rule.update_trace(&mut self.buffer, buffer)
    }

    /// Reset the trace to zeros.
    ///
    /// # Example
    ///
    /// ```
    /// use approx::assert_abs_diff_eq;
    /// use rsrl::{params::Vector, traces::{Trace, Accumulate}};
    ///
    /// let mut trace = Trace::new(Vector::ones(1), Accumulate {
    ///     gamma: 0.95,
    ///     lambda: 0.7,
    /// });
    ///
    /// assert_abs_diff_eq!(trace.buffer[0], 1.0);
    ///
    /// trace.reset();
    /// assert_abs_diff_eq!(trace.buffer[0], 0.0);
    /// ```
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

/// Trait for eligibility trace update rules.
pub trait UpdateRule<B: BufferMut> {
    /// Mutate the `trace` given a new `buffer` instance.
    ///
    /// # Arguments
    ///
    /// * `trace` - Mutable reference to the internal trace buffer.
    /// * `buffer` - New gradient buffer instance.
    fn update_trace(&self, trace: &mut B, buffer: &B);
}

/// Accumulating eligibility trace rule.
pub struct Accumulate {
    /// Discount factor.
    pub gamma: f64,

    /// Forgetting rate.
    pub lambda: f64,
}

impl<B: BufferMut> UpdateRule<B> for Accumulate {
    fn update_trace(&self, trace: &mut B, buffer: &B) {
        let rate = self.gamma * self.lambda;

        trace.merge_inplace(buffer, |x, y| rate * x + y)
    }
}

/// Replacing (saturating) eligibility trace rule.
pub struct Saturate {
    /// Discount factor.
    pub gamma: f64,

    /// Forgetting rate.
    pub lambda: f64,
}

impl<B: BufferMut> UpdateRule<B> for Saturate {
    fn update_trace(&self, trace: &mut B, buffer: &B) {
        let rate = self.gamma * self.lambda;

        trace
            .merge_inplace(buffer, |x, y| f64::max(-1.0, f64::min(1.0, rate * x + y)));
    }
}

/// Dutch eligibility trace rule.
pub struct Dutch {
    /// Learning rate.
    pub alpha: f64,

    /// Discount factor.
    pub gamma: f64,

    /// Forgetting rate.
    pub lambda: f64,
}

impl<B: BufferMut> UpdateRule<B> for Dutch {
    fn update_trace(&self, trace: &mut B, buffer: &B) {
        let rate = self.gamma * self.lambda * (1.0 - self.alpha);

        trace.merge_inplace(buffer, |x, y| rate * x + y)
    }
}
