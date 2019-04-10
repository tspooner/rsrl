use crate::{
    fa::{Approximator, Embedded, Parameterised, EvaluationResult, Features, UpdateResult},
    geometry::{Matrix, MatrixView, MatrixViewMut, Vector},
};
use ndarray::Axis;
use std::ops::MulAssign;

pub trait StdDev<I, M>: Approximator + Embedded<I> {
    fn stddev(&self, input: &I) -> Self::Output;

    fn grad_log(&self, input: &I, a: &M, mean: M) -> Vector<f64>;

    fn update_stddev(&mut self, input: &I, a: &M, mean: M, error: f64);
}

// Constant:
pub struct Constant<V: Clone>(pub V);

impl<V: Clone> Parameterised for Constant<V> {
    fn weights(&self) -> Matrix<f64> {
        Matrix::zeros((0, 0))
    }

    fn weights_view(&self) -> MatrixView<f64> {
        MatrixView::from_shape((0, 0), &[]).unwrap()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        MatrixViewMut::from_shape((0, 0), &mut []).unwrap()
    }
}

impl<I, V: Clone> Embedded<I> for Constant<V> {
    fn n_features(&self) -> usize {
        0
    }

    fn to_features(&self, _: &I) -> Features {
        Features::Dense(vec![].into())
    }
}

impl Approximator for Constant<f64> {
    type Output = f64;

    fn n_outputs(&self) -> usize { 1 }

    fn evaluate(&self, _: &Features) -> EvaluationResult<Self::Output> {
        Ok(self.0)
    }

    fn update(&mut self, _: &Features, _: Self::Output) -> UpdateResult<()> {
        Ok(())
    }
}

impl Approximator for Constant<(f64, f64)> {
    type Output = (f64, f64);

    fn n_outputs(&self) -> usize { 2 }

    fn evaluate(&self, _: &Features) -> EvaluationResult<Self::Output> {
        Ok(self.0)
    }

    fn update(&mut self, _: &Features, _: Self::Output) -> UpdateResult<()> {
        Ok(())
    }
}

impl Approximator for Constant<Vector<f64>> {
    type Output = Vector<f64>;

    fn n_outputs(&self) -> usize { self.0.len() }

    fn evaluate(&self, _: &Features) -> EvaluationResult<Self::Output> {
        Ok(self.0.clone())
    }

    fn update(&mut self, _: &Features, _: Self::Output) -> UpdateResult<()> {
        Ok(())
    }
}

impl Approximator for Constant<Matrix<f64>> {
    type Output = Matrix<f64>;

    fn n_outputs(&self) -> usize { self.0.len() }

    fn evaluate(&self, _: &Features) -> EvaluationResult<Self::Output> {
        Ok(self.0.clone())
    }

    fn update(&mut self, _: &Features, _: Self::Output) -> UpdateResult<()> {
        Ok(())
    }
}

impl<I, V: Clone, M> StdDev<I, M> for Constant<V>
where
    Self: Approximator<Output = V>
{
    fn stddev(&self, _: &I) -> V {
        self.0.clone()
    }

    fn grad_log(&self, _: &I, _: &M, _: M) -> Vector<f64> {
        Vector::zeros(0)
    }

    fn update_stddev(&mut self, _: &I, _: &M, _: M, _: f64) {}
}

// Scalar:
pub struct Scalar<F: Approximator<Output = f64>>(pub F);

impl_newtype_fa!(Scalar.0 => f64);

impl<I, F: Approximator<Output = f64> + Embedded<I>> StdDev<I, f64> for Scalar<F> {
    fn stddev(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.to_features(input)).unwrap()
    }

    fn grad_log(&self, input: &I, a: &Self::Output, mean: f64) -> Vector<f64> {
        let phi = self.to_features(input);
        let stddev = self.evaluate(&phi).unwrap();
        let phi = phi.expanded(self.n_features());

        let diff_sq = (a - mean).powi(2);

        (diff_sq / stddev - stddev / 2.0) * phi
    }

    fn update_stddev(&mut self, input: &I, a: &Self::Output, stddev: f64, error: f64) {
        self.update(&Features::Dense(self.grad_log(input, &a, stddev)), error).ok();
    }
}
