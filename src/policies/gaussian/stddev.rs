use crate::{
    fa::{Approximator, Embedding, Parameterised, EvaluationResult, Features, UpdateResult},
    geometry::{Matrix, MatrixView, MatrixViewMut, Vector},
};
use ndarray::Axis;
use std::ops::MulAssign;

const MIN_STDDEV: f64 = 0.05;

fn gl_from_mv(a: f64, mean: f64, stddev: f64) -> f64 {
    let diff_sq = (a - mean).powi(2);

    (diff_sq / stddev / stddev / stddev - 1.0 / stddev)
}

pub trait StdDev<I, M>: Approximator + Embedding<I> {
    fn stddev(&self, input: &I) -> Self::Output;

    fn grad_log(&self, input: &I, a: &M, mean: M) -> Matrix<f64>;

    fn update_stddev(&mut self, input: &I, a: &M, mean: M, error: f64);
}

// Constant:
#[derive(Clone, Debug, Serialize, Deserialize)]
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

impl<I, V: Clone> Embedding<I> for Constant<V> {
    fn n_features(&self) -> usize {
        0
    }

    fn embed(&self, _: &I) -> Features {
        Features::Dense(vec![].into())
    }
}

macro_rules! impl_approximator_constant {
    ($type:ty => $n:expr) => {
        impl Approximator for Constant<$type> {
            type Output = $type;

            fn n_outputs(&self) -> usize { $n }

            fn evaluate(&self, _: &Features) -> EvaluationResult<Self::Output> {
                Ok(self.0)
            }

            fn jacobian(&self, _: &Features) -> Matrix<f64> {
                Matrix::from_shape_vec((0, 0), vec![]).unwrap()
            }

            fn update_grad(&mut self, _: &Matrix<f64>, _: Self::Output) -> UpdateResult<()> {
                Ok(())
            }

            fn update(&mut self, _: &Features, _: Self::Output) -> UpdateResult<()> {
                Ok(())
            }
        }
    };
}

impl_approximator_constant!(f64 => 1);
impl_approximator_constant!([f64; 2] => 2);
impl_approximator_constant!([f64; 3] => 3);

impl Approximator for Constant<Vector<f64>> {
    type Output = Vector<f64>;

    fn n_outputs(&self) -> usize { self.0.len() }

    fn evaluate(&self, _: &Features) -> EvaluationResult<Self::Output> {
        Ok(self.0.clone())
    }

    fn jacobian(&self, _: &Features) -> Matrix<f64> {
        Matrix::from_shape_vec((0, 0), vec![]).unwrap()
    }

    fn update_grad(&mut self, _: &Matrix<f64>, _: Self::Output) -> UpdateResult<()> {
        Ok(())
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

    fn grad_log(&self, _: &I, _: &M, _: M) -> Matrix<f64> {
        Matrix::default((0, 0))
    }

    fn update_stddev(&mut self, _: &I, _: &M, _: M, _: f64) {}
}

// Scalar:
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scalar<F: Approximator<Output = f64>>(pub F);

impl_newtype_fa!(Scalar.0 => f64);

impl<I, F: Approximator<Output = f64> + Embedding<I>> StdDev<I, f64> for Scalar<F> {
    fn stddev(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.embed(input)).unwrap() + MIN_STDDEV
    }

    fn grad_log(&self, input: &I, a: &f64, mean: f64) -> Matrix<f64> {
        let phi = self.embed(input);
        let stddev = self.evaluate(&phi).unwrap() + MIN_STDDEV;
        let gl_partial = gl_from_mv(*a, mean, stddev);

        (phi.expanded(self.0.n_features()) * gl_partial).insert_axis(Axis(1))
    }

    fn update_stddev(&mut self, input: &I, a: &f64, mean: f64, error: f64) {
        let phi = self.embed(input);
        let stddev = self.evaluate(&phi).unwrap() + MIN_STDDEV;

        self.update(&phi, gl_from_mv(*a, mean, stddev) * error).ok();
    }
}

// Pair:
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pair<F: Approximator<Output = [f64; 2]>>(pub F);

impl_newtype_fa!(Pair.0 => [f64; 2]);

impl<I, F: Approximator<Output = [f64; 2]> + Embedding<I>> StdDev<I, [f64; 2]> for Pair<F> {
    fn stddev(&self, input: &I) -> Self::Output {
        let raw = self.0.evaluate(&self.0.embed(input)).unwrap();

        [raw[0] + MIN_STDDEV, raw[1] + MIN_STDDEV]
    }

    fn grad_log(&self, input: &I, a: &[f64; 2], mean: [f64; 2]) -> Matrix<f64> {
        let phi = self.embed(input);
        let stddev = self.evaluate(&phi).unwrap();

        let n_features = self.0.n_features();
        let phi = phi.expanded(n_features);

        let gl_partial_0 = gl_from_mv(a[0], mean[0], stddev[0] + MIN_STDDEV);
        let gl_partial_1 = gl_from_mv(a[1], mean[1], stddev[1] + MIN_STDDEV);

        Vector::from_iter(
            phi.iter().map(|v| v * gl_partial_0).chain(phi.iter().map(|v| v * gl_partial_1))
        ).into_shape((2, n_features)).unwrap().reversed_axes()
    }

    fn update_stddev(&mut self, input: &I, a: &[f64; 2], mean: [f64; 2], error: f64) {
        let phi = self.embed(input);
        let stddev = self.evaluate(&phi).unwrap();

        self.update(&phi, [
            gl_from_mv(a[0], mean[0], stddev[0] + MIN_STDDEV) * error,
            gl_from_mv(a[1], mean[1], stddev[1] + MIN_STDDEV) * error
        ]).ok();
    }
}
