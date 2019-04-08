use crate::{
    core::{Algorithm},
    fa::{Approximator, Parameterised},
    geometry::{Matrix, MatrixView, MatrixViewMut},
    policies::{DifferentiablePolicy, ParameterisedPolicy, Policy},
};
use ndarray::Axis;
use super::UnivariateGaussian;

pub struct SphericalMultivariateGaussian<F>(Vec<UnivariateGaussian<F>>);

impl<F> SphericalMultivariateGaussian<F> {
    pub fn new(policies: Vec<UnivariateGaussian<F>>) -> Self {
        SphericalMultivariateGaussian(policies)
    }
}

impl<F> Algorithm for SphericalMultivariateGaussian<F> {
    fn handle_terminal(&mut self) {
        for p in self.0.iter_mut() {
            p.handle_terminal();
        }
    }
}

impl<S, F> Policy<S> for SphericalMultivariateGaussian<F>
where
    UnivariateGaussian<F>: Policy<S>,
{
    type Action = Vec<<UnivariateGaussian<F> as Policy<S>>::Action>;

    fn sample(&mut self, s: &S) -> Self::Action {
        self.0.iter_mut().map(|p| p.sample(s)).collect()
    }

    fn mpa(&mut self, s: &S) -> Self::Action {
        self.0.iter_mut().map(|p| p.mpa(s)).collect()
    }

    fn probability(&mut self, s: &S, a: Self::Action) -> f64 {
        self.0.iter_mut().zip(a.into_iter()).map(|(p, a)| p.probability(s, a)).product()
    }
}

impl<S, F> DifferentiablePolicy<S> for SphericalMultivariateGaussian<F>
where
    UnivariateGaussian<F>: DifferentiablePolicy<S> + Parameterised,
{
    fn grad_log(&self, input: &S, a: Self::Action) -> Matrix<f64> {
        let mut g = unsafe { Matrix::uninitialized(self.weights_dim()) };

        for (i, (p, a)) in self.0.iter().zip(a.into_iter()).enumerate() {
            g.column_mut(i).assign(&p.grad_log(input, a));
        }

        g
    }
}

impl<F> Parameterised for SphericalMultivariateGaussian<F>
where
    UnivariateGaussian<F>: Parameterised,
{
    fn weights(&self) -> Matrix<f64> {
        let mut w = unsafe { Matrix::uninitialized(self.weights_dim()) };

        for (i, p) in self.0.iter().enumerate() {
            w.column_mut(i).assign(&p.weights());
        }

        w
    }

    fn weights_view(&self) -> MatrixView<f64> {
        unimplemented!()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }

    fn weights_count(&self) -> usize {
        self.0.iter().map(|p| p.weights_count()).sum()
    }

    fn weights_dim(&self) -> (usize, usize) {
        (self.0[0].weights_count(), self.0.len())
    }
}

impl<S, F> ParameterisedPolicy<S> for SphericalMultivariateGaussian<F>
where
    UnivariateGaussian<F>: ParameterisedPolicy<S>,
{
    fn update(&mut self, input: &S, a: Self::Action, error: f64) {
        self.0.iter_mut().zip(a.into_iter()).for_each(|(p, a)| {
            p.update(input, a, error)
        });
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.0.iter_mut().zip(errors.gencolumns().into_iter()).for_each(|(p, c)| {
            p.update_raw(c.into_owned().insert_axis(Axis(1)))
        });
    }
}
