use crate::{
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction, DifferentiableStateFunction,
    },
    geometry::{Matrix, MatrixView, MatrixViewMut},
};
use std::ops::MulAssign;

const MIN_STDDEV: f64 = 0.1;
const STDDEV_TOL: f64 = 0.2;

fn gl_from_mv(a: f64, mean: f64, stddev: f64) -> f64 {
    let diff_sq = (a - mean).powi(2);
    let stddev = stddev.max(STDDEV_TOL);

    (diff_sq / stddev / stddev / stddev - 1.0 / stddev)
}

pub trait StdDev<I, M>: StateFunction<I> + Parameterised {
    fn stddev(&self, input: &I) -> Self::Output;

    fn grad_log(&self, input: &I, a: &M, mean: M) -> Matrix<f64>;

    fn update_stddev(&mut self, input: &I, a: &M, mean: M, error: f64);
}

// Constant:
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Constant<V>(pub V);

impl<I, V: Clone> StateFunction<I> for Constant<V> {
    type Output = V;

    fn evaluate(&self, _: &I) -> Self::Output { self.0.clone() }

    fn update(&mut self, _: &I, _: Self::Output) {}
}

impl<V> Parameterised for Constant<V> {
    fn weights(&self) -> Matrix<f64> { Matrix::zeros((0, 0)) }

    fn weights_view(&self) -> MatrixView<f64> { MatrixView::from_shape((0, 0), &[]).unwrap() }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        MatrixViewMut::from_shape((0, 0), &mut []).unwrap()
    }
}

impl<I, V: Clone, M> StdDev<I, M> for Constant<V> {
    fn stddev(&self, _: &I) -> V { self.0.clone() }

    fn grad_log(&self, _: &I, _: &M, _: M) -> Matrix<f64> { Matrix::default((0, 0)) }

    fn update_stddev(&mut self, _: &I, _: &M, _: M, _: f64) {}
}

// Scalar:
#[derive(Clone, Debug, Serialize, Deserialize, Parameterised)]
pub struct Scalar<F>(pub F);

impl<I, F> StateFunction<I> for Scalar<F>
where
    F: StateFunction<I, Output = f64>,
{
    type Output = f64;

    fn evaluate(&self, state: &I) -> Self::Output {
        (self.0.evaluate(state) + MIN_STDDEV).max(0.0)
    }

    fn update(&mut self, state: &I, error: Self::Output) { self.0.update(state, error) }
}

impl<I, F> StdDev<I, f64> for Scalar<F>
where
    F: DifferentiableStateFunction<I, Output = f64> + Parameterised,
{
    fn stddev(&self, input: &I) -> Self::Output {
        self.evaluate(input)
    }

    fn grad_log(&self, input: &I, a: &f64, mean: f64) -> Matrix<f64> {
        self.0.grad(input).into() * gl_from_mv(*a, mean, self.evaluate(input))
    }

    fn update_stddev(&mut self, input: &I, a: &f64, mean: f64, error: f64) {
        let stddev = self.evaluate(input);

        self.update(input, gl_from_mv(*a, mean, stddev) * error);
    }
}

// Pair:
#[derive(Clone, Debug, Serialize, Deserialize, Parameterised)]
pub struct Pair<F>(pub F);

impl<I, F> StateFunction<I> for Pair<F>
where
    F: StateFunction<I, Output = [f64; 2]>,
{
    type Output = [f64; 2];

    fn evaluate(&self, state: &I) -> Self::Output {
        let raw = self.0.evaluate(state);

        [
            (raw[0] + MIN_STDDEV).max(0.0),
            (raw[1] + MIN_STDDEV).max(0.0)
        ]
    }

    fn update(&mut self, state: &I, error: Self::Output) { self.0.update(state, error) }
}

impl<I, F> StdDev<I, [f64; 2]> for Pair<F>
where
    F: DifferentiableStateFunction<I, Output = [f64; 2]> + Parameterised,
{
    fn stddev(&self, input: &I) -> Self::Output {
        self.evaluate(input)
    }

    fn grad_log(&self, input: &I, a: &[f64; 2], mean: [f64; 2]) -> Matrix<f64> {
        let mut g = self.0.grad(input).into();
        let stddev = self.evaluate(input);

        g.column_mut(0).mul_assign(gl_from_mv(a[0], mean[0], stddev[0]));
        g.column_mut(0).mul_assign(gl_from_mv(a[1], mean[1], stddev[1]));

        g
    }

    fn update_stddev(&mut self, input: &I, a: &[f64; 2], mean: [f64; 2], error: f64) {
        let stddev = self.evaluate(input);

        self.update(
            input,
            [
                gl_from_mv(a[0], mean[0], stddev[0]) * error,
                gl_from_mv(a[1], mean[1], stddev[1]) * error,
            ],
        );
    }
}
