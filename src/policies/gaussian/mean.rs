use crate::{
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction, DifferentiableStateFunction,
    },
    utils::pinv,
};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use std::ops::MulAssign;

const STDDEV_TOL: f64 = 0.2;

pub trait Mean<I, S>: StateFunction<I> + Parameterised {
    fn mean(&self, input: &I) -> Self::Output;

    fn grad_log(&self, input: &I, a: &Self::Output, stddev: S) -> Array2<f64>;

    fn update_mean(&mut self, input: &I, a: &Self::Output, stddev: S, error: f64);
}

// Scalar:
#[derive(Clone, Debug, Serialize, Deserialize, Parameterised)]
pub struct Scalar<F>(pub F);

impl<I, F> StateFunction<I> for Scalar<F>
where
    F: StateFunction<I, Output = f64>,
{
    type Output = f64;

    fn evaluate(&self, state: &I) -> Self::Output { self.0.evaluate(state) }

    fn update(&mut self, state: &I, error: Self::Output) { self.0.update(state, error) }
}

impl<I, F> Mean<I, f64> for Scalar<F>
where
    F: DifferentiableStateFunction<I, Output = f64> + Parameterised,
{
    fn mean(&self, input: &I) -> Self::Output { self.0.evaluate(input) }

    fn grad_log(&self, input: &I, a: &f64, stddev: f64) -> Array2<f64> {
        let mean = self.evaluate(input);
        let stddev = stddev.max(STDDEV_TOL);

        self.0.grad(input).into() * (a - mean) / (stddev * stddev)
    }

    fn update_mean(&mut self, input: &I, a: &f64, stddev: f64, error: f64) {
        let mean = self.evaluate(input);
        let stddev = stddev.max(STDDEV_TOL);

        self.update(input, (a - mean) / (stddev * stddev) * error);
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

    fn evaluate(&self, state: &I) -> Self::Output { self.0.evaluate(state) }

    fn update(&mut self, state: &I, error: Self::Output) { self.0.update(state, error) }
}

impl<I, F> Mean<I, f64> for Pair<F>
where
    F: DifferentiableStateFunction<I, Output = [f64; 2]> + Parameterised,
{
    fn mean(&self, input: &I) -> Self::Output { self.0.evaluate(input) }

    fn grad_log(&self, input: &I, actions: &[f64; 2], stddev: f64) -> Array2<f64> {
        // (2 x 1)
        let means = self.evaluate(input);
        let stddev = stddev.max(STDDEV_TOL);

        // (N x 2)
        let mut g = self.0.grad(input).into();

        g.column_mut(0).mul_assign((actions[0] - means[0]) / (stddev * stddev));
        g.column_mut(1).mul_assign((actions[1] - means[1]) / (stddev * stddev));

        g
    }

    fn update_mean(&mut self, input: &I, actions: &[f64; 2], stddev: f64, error: f64) {
        let means = self.evaluate(input);
        let stddev = stddev.max(STDDEV_TOL);

        self.update(input, [
            (actions[0] - means[0]) / (stddev * stddev) * error,
            (actions[1] - means[1]) / (stddev * stddev) * error,
        ]);
    }
}

impl<I, F> Mean<I, [f64; 2]> for Pair<F>
where
    F: DifferentiableStateFunction<I, Output = [f64; 2]> + Parameterised,
{
    fn mean(&self, input: &I) -> Self::Output { self.0.evaluate(input) }

    fn grad_log(&self, input: &I, actions: &[f64; 2], stddev: [f64; 2]) -> Array2<f64> {
        // (2 x 1)
        let means = self.evaluate(input);
        let stddev = [stddev[0].max(STDDEV_TOL), stddev[1].max(STDDEV_TOL)];

        // (N x 2)
        let mut g = self.0.grad(input).into();

        g.column_mut(0).mul_assign((actions[0] - means[0]) / (stddev[0] * stddev[0]));
        g.column_mut(1).mul_assign((actions[1] - means[1]) / (stddev[1] * stddev[1]));

        g
    }

    fn update_mean(&mut self, input: &I, actions: &[f64; 2], stddev: [f64; 2], error: f64) {
        let means = self.evaluate(input);
        let stddev = [stddev[0].max(STDDEV_TOL), stddev[1].max(STDDEV_TOL)];

        self.update(input, [
            (actions[0] - means[0]) / (stddev[0] * stddev[0]) * error,
            (actions[1] - means[1]) / (stddev[1] * stddev[1]) * error,
        ]);
    }
}

// Multi:
#[derive(Clone, Debug, Serialize, Deserialize, Parameterised)]
pub struct Multi<F>(pub F);

impl<I, F> StateFunction<I> for Multi<F>
where
    F: StateFunction<I, Output = Vec<f64>>,
{
    type Output = Vec<f64>;

    fn evaluate(&self, state: &I) -> Self::Output { self.0.evaluate(state) }

    fn update(&mut self, state: &I, error: Self::Output) { self.0.update(state, error) }
}

impl<F> Multi<F> {
    fn gl_fmv_partial<I>(
        &self,
        input: &I,
        actions: &[f64],
        sigma: Array2<f64>,
    ) -> Array2<f64>
    where
        F: StateFunction<I, Output = Vec<f64>>,
    {
        // A x 1
        let a_diff = Array1::from_iter(
            self.evaluate(input).into_iter().zip(actions.iter()).map(|(m, a)| a - m)
        );

        // A x A
        let sigma_inv = pinv(&sigma).unwrap();

        // [(1 x A) . (A x A)]^T => (A x 1)
        a_diff.insert_axis(Axis(0)).dot(&sigma_inv)
    }
}

impl<I, F> Mean<I, f64> for Multi<F>
where
    F: DifferentiableStateFunction<I, Output = Vec<f64>> + Parameterised,
{
    fn mean(&self, input: &I) -> Self::Output { self.0.evaluate(input) }

    fn grad_log(&self, input: &I, actions: &Vec<f64>, stddev: f64) -> Array2<f64> {
        let actions = unsafe { ArrayView1::from_shape_ptr(actions.len(), actions.as_ptr()) };

        // (A x 1)
        let means = Array1::from_vec(self.evaluate(input));
        let stddev = stddev.max(STDDEV_TOL);

        // (A x 1)
        let gl_partial = (means - actions).mapv_into(|v| -v / (stddev * stddev));

        // (N x A)
        let jacobian = self.0.grad(input).into();

        jacobian * gl_partial.t()
    }

    fn update_mean(&mut self, input: &I, actions: &Vec<f64>, stddev: f64, error: f64) {
        let means = self.evaluate(input);
        let stddev = stddev.max(STDDEV_TOL);
        let var = stddev * stddev;

        let updates = means.into_iter().zip(actions.iter()).map(|(m, a)| (a - m) / var * error);

        self.update(input, updates.collect());
    }
}

impl<I, F> Mean<I, Vec<f64>> for Multi<F>
where
    F: DifferentiableStateFunction<I, Output = Vec<f64>> + Parameterised,
{
    fn mean(&self, input: &I) -> Self::Output { self.0.evaluate(input) }

    fn grad_log(&self, input: &I, actions: &Vec<f64>, stddev: Vec<f64>) -> Array2<f64> {
        // (A x 1)
        let means = self.evaluate(input);

        // (A x 1)
        let gl_partial = means
            .into_iter()
            .zip(actions.into_iter().zip(stddev.into_iter()))
            .map(|(m, (a, s))| {
                let s = s.max(STDDEV_TOL);

                (a - m) / (s * s)
            })
            .collect::<Array1<f64>>()
            .insert_axis(Axis(1));

        // (N x A)
        let jacobian = self.0.grad(input).into();

        jacobian * gl_partial.t()
    }

    fn update_mean(&mut self, input: &I, actions: &Vec<f64>, stddev: Vec<f64>, error: f64) {
        let means = self.evaluate(input);

        let gl_partial_scaled = means
            .into_iter()
            .zip(actions.into_iter().zip(stddev.into_iter()))
            .map(|(m, (a, s))| {
                let s = s.max(STDDEV_TOL);

                (a - m) / (s * s) * error
            })
            .collect();

        self.update(input, gl_partial_scaled);
    }
}

impl<I, F> Mean<I, Array2<f64>> for Multi<F>
where
    F: DifferentiableStateFunction<I, Output = Vec<f64>> + Parameterised,
{
    fn mean(&self, input: &I) -> Self::Output { self.0.evaluate(input) }

    fn grad_log(&self, input: &I, actions: &Vec<f64>, sigma: Array2<f64>) -> Array2<f64> {
        // (A x 1)
        let gl_partial = self.gl_fmv_partial(input, actions, sigma);

        // (N x A)
        let jacobian = self.0.grad(input).into();

        jacobian * gl_partial.t()
    }

    fn update_mean(&mut self, input: &I, actions: &Vec<f64>, sigma: Array2<f64>, error: f64) {
        let gl_partial = self.gl_fmv_partial(input, actions, sigma);

        self.update(input, (gl_partial * error).into_raw_vec());
    }
}
