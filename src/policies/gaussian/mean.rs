use crate::{
    fa::{self, Approximator, Embedding, Parameterised, EvaluationResult, Features, UpdateResult},
    geometry::{Matrix, MatrixView, MatrixViewMut, Vector},
    utils::pinv,
};
use ndarray::Axis;
use std::ops::MulAssign;

pub trait Mean<I, S>: Approximator + Embedding<I> {
    fn mean(&self, input: &I) -> Self::Output;

    fn grad_log(&self, input: &I, a: &Self::Output, stddev: S) -> Matrix<f64>;

    fn update_mean(&mut self, input: &I, a: &Self::Output, stddev: S, error: f64);
}

// Scalar:
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scalar<F: Approximator<Output = f64>>(pub F);

impl_newtype_fa!(Scalar.0 => f64);

impl<I, F: Approximator<Output = f64> + Embedding<I>> Mean<I, f64> for Scalar<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.embed(input)).unwrap()
    }

    fn grad_log(&self, input: &I, a: &f64, stddev: f64) -> Matrix<f64> {
        let phi = self.0.embed(input);
        let mean = self.evaluate(&phi).unwrap();
        let gl_partial = (a - mean) / stddev / stddev;

        self.0.jacobian(&phi) * gl_partial
    }

    fn update_mean(&mut self, input: &I, a: &f64, stddev: f64, error: f64) {
        let phi = self.0.embed(input);
        let mean = self.evaluate(&phi).unwrap();

        self.update(&phi, (a - mean) / stddev / stddev * error).ok();
    }
}

// Pair:
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pair<F: Approximator<Output = [f64; 2]>>(pub F);

impl_newtype_fa!(Pair.0 => [f64; 2]);

impl<I, F: Approximator<Output = [f64; 2]> + Embedding<I>> Mean<I, f64> for Pair<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.embed(input)).unwrap()
    }

    fn grad_log(&self, input: &I, actions: &[f64; 2], stddev: f64) -> Matrix<f64> {
        let phi = self.0.embed(input);

        // (2 x 1)
        let means = self.evaluate(&phi).unwrap();

        // (N x 2)
        let mut g = self.0.jacobian(&phi);

        g.column_mut(0).mul_assign((actions[0] - means[0]) / stddev / stddev);
        g.column_mut(1).mul_assign((actions[1] - means[1]) / stddev / stddev);

        g
    }

    fn update_mean(&mut self, input: &I, actions: &[f64; 2], stddev: f64, error: f64) {
        let phi = self.0.embed(input);
        let means = self.evaluate(&phi).unwrap();

        self.update(&phi, [
            (actions[0] - means[0]) / stddev / stddev * error,
            (actions[1] - means[1]) / stddev / stddev * error
        ]).ok();
    }
}

impl<I, F: Approximator<Output = [f64; 2]> + Embedding<I>> Mean<I, [f64; 2]> for Pair<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.embed(input)).unwrap()
    }

    fn grad_log(&self, input: &I, actions: &[f64; 2], stddev: [f64; 2]) -> Matrix<f64> {
        let phi = self.0.embed(input);

        // (2 x 1)
        let means = self.evaluate(&phi).unwrap();

        // (N x 2)
        let mut g = self.0.jacobian(&phi);

        g.column_mut(0).mul_assign((actions[0] - means[0]) / stddev[0] / stddev[0]);
        g.column_mut(1).mul_assign((actions[1] - means[1]) / stddev[1] / stddev[1]);

        g
    }

    fn update_mean(&mut self, input: &I, actions: &[f64; 2], stddev: [f64; 2], error: f64) {
        let phi = self.0.embed(input);
        let means = self.evaluate(&phi).unwrap();

        self.update(&phi, [
            (actions[0] - means[0]) / stddev[0] / stddev[0] * error,
            (actions[1] - means[1]) / stddev[1] / stddev[1] * error
        ]).ok();
    }
}

// Multi:
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Multi<F: Approximator<Output = Vector<f64>>>(pub F);

impl_newtype_fa!(Multi.0 => Vector<f64>);

impl<F: Approximator<Output = Vector<f64>>> Multi<F> {
    fn gl_fmv_partial(&self, phi: &Features, actions: &Vector<f64>, sigma: Matrix<f64>) -> Matrix<f64> {
        let means = self.evaluate(&phi).unwrap();

        // A x 1
        let a_diff = -(means - actions);

        // A x A
        let sigma_inv = pinv(&sigma).unwrap();

        // [(1 x A) . (A x A)]^T => (A x 1)
        a_diff.insert_axis(Axis(0)).dot(&sigma_inv)
    }
}

impl<I, F: Approximator<Output = Vector<f64>> + Embedding<I>> Mean<I, f64> for Multi<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.embed(input)).unwrap()
    }

    fn grad_log(&self, input: &I, actions: &Vector<f64>, stddev: f64) -> Matrix<f64> {
        let phi = self.0.embed(input);

        // (A x 1)
        let means = self.evaluate(&phi).unwrap();

        // (A x 1)
        let gl_partial = (means - actions).mapv_into(|v| -v / stddev / stddev);

        // (N x A)
        let jacobian = self.0.jacobian(&phi);

        jacobian * gl_partial.t()
    }

    fn update_mean(&mut self, input: &I, actions: &Vector<f64>, stddev: f64, error: f64) {
        let phi = self.0.embed(input);
        let means = self.evaluate(&phi).unwrap();

        let gl_partial_scaled = (means - actions).mapv_into(|v| -v / stddev / stddev * error);

        self.update(&phi, gl_partial_scaled).ok();
    }
}

impl<I, F: Approximator<Output = Vector<f64>> + Embedding<I>> Mean<I, Vector<f64>> for Multi<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.embed(input)).unwrap()
    }

    fn grad_log(&self, input: &I, actions: &Vector<f64>, stddev: Vector<f64>) -> Matrix<f64> {
        let phi = self.0.embed(input);

        // (A x 1)
        let means = self.evaluate(&phi).unwrap();

        // (A x 1)
        let gl_partial = means
            .into_iter()
            .zip(actions.into_iter().zip(stddev.into_iter()))
            .map(|(m, (a, s))| (a - m) / s / s)
            .collect::<Vector<f64>>()
            .insert_axis(Axis(1));

        // (N x A)
        let jacobian = self.0.jacobian(&phi);

        jacobian * gl_partial.t()
    }

    fn update_mean(&mut self, input: &I, actions: &Vector<f64>, stddev: Vector<f64>, error: f64) {
        let phi = self.0.embed(input);
        let means = self.evaluate(&phi).unwrap();

        let gl_partial_scaled = means
            .into_iter()
            .zip(actions.into_iter().zip(stddev.into_iter()))
            .map(|(m, (a, s))| (a - m) / s / s * error)
            .collect();

        self.update(&phi, gl_partial_scaled).ok();
    }
}

impl<I, F: Approximator<Output = Vector<f64>> + Embedding<I>> Mean<I, Matrix<f64>> for Multi<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.embed(input)).unwrap()
    }

    fn grad_log(&self, input: &I, actions: &Vector<f64>, sigma: Matrix<f64>) -> Matrix<f64> {
        // (N x 1)
        let phi = self.embed(input);

        // (A x 1)
        let gl_partial = self.gl_fmv_partial(&phi, actions, sigma);

        // (N x A)
        let jacobian = self.0.jacobian(&phi);

        jacobian * gl_partial.t()
    }

    fn update_mean(&mut self, input: &I, actions: &Vector<f64>, sigma: Matrix<f64>, error: f64) {
        let phi = self.embed(input);
        let gl_partial = self.gl_fmv_partial(&phi, actions, sigma).index_axis_move(Axis(1), 0);

        self.update(&phi, gl_partial * error).ok();
    }
}
