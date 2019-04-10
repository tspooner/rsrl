use crate::{
    fa::{self, Approximator, Embedded, Parameterised, EvaluationResult, Features, UpdateResult},
    geometry::{Matrix, MatrixView, MatrixViewMut, Vector},
    utils::pinv,
};
use ndarray::Axis;
use std::ops::MulAssign;

pub trait Mean<I, S>: Approximator + Embedded<I> {
    fn mean(&self, input: &I) -> Self::Output;

    fn grad_log(&self, input: &I, a: &Self::Output, stddev: S) -> Vector<f64>;

    fn update_mean(&mut self, input: &I, a: &Self::Output, stddev: S, error: f64);
}

// Scalar:
pub struct Scalar<F: Approximator<Output = f64>>(pub F);

impl_newtype_fa!(Scalar.0 => f64);

impl<I, F: Approximator<Output = f64> + Embedded<I>> Mean<I, f64> for Scalar<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.to_features(input)).unwrap()
    }

    fn grad_log(&self, input: &I, a: &Self::Output, _: f64) -> Vector<f64> {
        let phi = self.to_features(input);
        let mean = self.evaluate(&phi).unwrap();
        let phi = phi.expanded(self.n_features());

        // (a - mean) / std / std * phi
        (a - mean) * phi
    }

    fn update_mean(&mut self, input: &I, a: &Self::Output, _: f64, error: f64) {
        let phi = self.to_features(input);
        let mean = self.evaluate(&phi).unwrap();

        self.update(&phi, (a - mean) * error).ok();
    }
}

// Pair:
pub struct Pair<F: Approximator<Output = (f64, f64)>>(pub F);

impl_newtype_fa!(Pair.0 => (f64, f64));

impl<F: Approximator<Output = (f64, f64)>> Pair<F> {
    fn grad_log_rescaled<I>(&self, input: &I, actions: &(f64, f64)) -> Vector<f64>
        where F: Embedded<I>
    {
        let nf = self.n_features();
        let phi = self.to_features(input);
        let means = self.evaluate(&phi).unwrap();

        let mut g = Vector::from_iter(phi.expanded(nf).to_vec().into_iter().cycle().take(nf * 2));

        g.slice_mut(s![0..nf]).mul_assign(actions.0 - means.0);
        g.slice_mut(s![nf..2*nf]).mul_assign(actions.1 - means.1);

        g
    }
}

impl<I, F: Approximator<Output = (f64, f64)> + Embedded<I>> Mean<I, f64> for Pair<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.to_features(input)).unwrap()
    }

    fn grad_log(&self, input: &I, actions: &(f64, f64), _: f64) -> Vector<f64> {
        // g.column_mut(0).div_assign(std * std);
        // g.column_mut(1).div_assign(std * std);

        self.grad_log_rescaled(input, actions)
    }

    fn update_mean(&mut self, input: &I, a: &Self::Output, stddev: f64, error: f64) {
        self.update(&Features::Dense(self.grad_log(input, &a, stddev)), (error, error)).ok();
    }
}

impl<I, F: Approximator<Output = (f64, f64)> + Embedded<I>> Mean<I, (f64, f64)> for Pair<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.to_features(input)).unwrap()
    }

    fn grad_log(&self, input: &I, actions: &(f64, f64), _: (f64, f64)) -> Vector<f64> {
        // g.column_mut(0).div_assign(std.0 * std.0);
        // g.column_mut(1).div_assign(std.1 * std.1);

        self.grad_log_rescaled(input, actions)
    }

    fn update_mean(&mut self, input: &I, a: &Self::Output, stddev: (f64, f64), error: f64) {
        self.update(&Features::Dense(self.grad_log(input, &a, stddev)), (error, error)).ok();
    }
}

// Multi:
pub struct Multi<F: Approximator<Output = Vector<f64>>>(pub F);

impl_newtype_fa!(Multi.0 => Vector<f64>);

impl<F: Approximator<Output = Vector<f64>>> Multi<F> {
    fn grad_log_rescaled<I>(&self, input: &I, actions: &Vector<f64>) -> Vector<f64>
        where F: Embedded<I>
    {
        let no = self.n_outputs();
        let nf = self.n_features();

        let phi = self.to_features(input);
        let means = self.evaluate(&phi).unwrap();

        let mut g = Vector::from_iter(phi.expanded(nf).to_vec().into_iter().cycle().take(nf * no));

        for (i, (a, m)) in actions.into_iter().zip(means.into_iter()).enumerate() {
            g.slice_mut(s![(i * nf)..((i + 1) * nf)]).mul_assign(a - m);
        }

        g
    }
}

impl<I, F: Approximator<Output = Vector<f64>> + Embedded<I>> Mean<I, f64> for Multi<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.to_features(input)).unwrap()
    }

    fn grad_log(&self, input: &I, actions: &Vector<f64>, _: f64) -> Vector<f64> {
        // for (i, (a, m)) in actions.into_iter().zip(means.into_iter()).enumerate() {
            // g.column_mut(i).mul_assign((a - m) / std / std);
        // }

        self.grad_log_rescaled(input, actions)
    }

    fn update_mean(&mut self, input: &I, a: &Self::Output, stddev: f64, error: f64) {
        let error = Vector::from_elem(self.n_outputs(), error);

        self.update(&Features::Dense(self.grad_log(input, &a, stddev)), error).ok();
    }
}

impl<I, F: Approximator<Output = Vector<f64>> + Embedded<I>> Mean<I, Vector<f64>> for Multi<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.to_features(input)).unwrap()
    }

    fn grad_log(&self, input: &I, actions: &Vector<f64>, _: Vector<f64>) -> Vector<f64> {
        // let ms = means.into_iter().zip(stddevs.into_iter());
        // for (i, (a, (m, s))) in actions.into_iter().zip(ms).enumerate() {
            // g.column_mut(i).mul_assign((a - m) / s / s);
        // }

        self.grad_log_rescaled(input, actions)
    }

    fn update_mean(&mut self, input: &I, a: &Self::Output, stddev: Vector<f64>, error: f64) {
        let error = Vector::from_elem(self.n_outputs(), error);

        self.update(&Features::Dense(self.grad_log(input, &a, stddev)), error).ok();
    }
}

impl<I, F: Approximator<Output = Vector<f64>> + Embedded<I>> Mean<I, Matrix<f64>> for Multi<F> {
    fn mean(&self, input: &I) -> Self::Output {
        self.0.evaluate(&self.0.to_features(input)).unwrap()
    }

    fn grad_log(&self, input: &I, actions: &Vector<f64>, sigma: Matrix<f64>) -> Vector<f64> {
        let no = self.n_outputs();
        let nf = self.n_features();

        let phi = self.to_features(input);
        let means = self.evaluate(&phi).unwrap();

        // N x 1
        let a_diff = -(means - actions);

        // N x N
        let sigma_inv = pinv(&sigma).unwrap();

        // (1 x N) . (N x N) => (1 x N)
        let grad_partial = a_diff.insert_axis(Axis(0)).dot(&sigma_inv);

        // (N*O x 1)
        let mut g = Vector::from_iter(phi.expanded(nf).to_vec().into_iter().cycle().take(nf * no));

        for (i, &gp) in grad_partial.into_iter().enumerate() {
            g.slice_mut(s![(i * nf)..((i + 1) * nf)]).mul_assign(gp);
        }

        g
    }

    fn update_mean(&mut self, input: &I, a: &Self::Output, stddev: Matrix<f64>, error: f64) {
        let error = Vector::from_elem(self.n_outputs(), error);

        self.update(&Features::Dense(self.grad_log(input, &a, stddev)), error).ok();
    }
}
