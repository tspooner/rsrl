use crate::{
    core::{Algorithm},
    fa::{Approximator, Parameterised},
    geometry::{Matrix, MatrixView, MatrixViewMut},
    policies::{DifferentiablePolicy, ParameterisedPolicy, Policy},
};
use ndarray::Axis;
use super::UnivariateGaussian;

pub struct SphericalBivariateGaussian<F>(UnivariateGaussian<F>, UnivariateGaussian<F>);

impl<F> SphericalBivariateGaussian<F> {
    pub fn new(p1: UnivariateGaussian<F>, p2: UnivariateGaussian<F>) -> Self {
        SphericalBivariateGaussian(p1, p2)
    }
}

impl<F> Algorithm for SphericalBivariateGaussian<F> {
    fn handle_terminal(&mut self) {
        self.0.handle_terminal();
        self.1.handle_terminal();
    }
}

impl<S, F> Policy<S> for SphericalBivariateGaussian<F>
where
    UnivariateGaussian<F>: Policy<S>,
{
    type Action = (
        <UnivariateGaussian<F> as Policy<S>>::Action,
        <UnivariateGaussian<F> as Policy<S>>::Action
    );

    fn sample(&mut self, s: &S) -> Self::Action {
        (self.0.sample(s), self.1.sample(s))
    }

    fn mpa(&mut self, s: &S) -> Self::Action {
        (self.0.mpa(s), self.1.mpa(s))
    }

    fn probability(&mut self, s: &S, a: Self::Action) -> f64 {
        self.0.probability(s, a.0) * self.1.probability(s, a.1)
    }
}

impl<S, F> DifferentiablePolicy<S> for SphericalBivariateGaussian<F>
where
    UnivariateGaussian<F>: DifferentiablePolicy<S>,
{
    fn grad_log(&self, input: &S, a: Self::Action) -> Matrix<f64> {
        stack![Axis(1), self.0.grad_log(input, a.0), self.1.grad_log(input, a.1)]
    }
}

impl<F> Parameterised for SphericalBivariateGaussian<F>
where
    UnivariateGaussian<F>: Parameterised,
{
    fn weights(&self) -> Matrix<f64> {
        stack![Axis(1), self.0.weights(), self.1.weights()]
    }

    fn weights_view(&self) -> MatrixView<f64> {
        unimplemented!()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }

    fn weights_count(&self) -> usize {
        self.0.weights_count() + self.1.weights_count()
    }

    fn weights_dim(&self) -> (usize, usize) {
        (self.0.weights_count(), 2)
    }
}

impl<S, F> ParameterisedPolicy<S> for SphericalBivariateGaussian<F>
where
    UnivariateGaussian<F>: ParameterisedPolicy<S>,
{
    fn update(&mut self, input: &S, a: Self::Action, error: f64) {
        self.0.update(input, a.0, error);
        self.1.update(input, a.1, error);
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.0.update_raw(errors.column(0).into_owned().insert_axis(Axis(1)));
        self.1.update_raw(errors.column(1).into_owned().insert_axis(Axis(1)));
    }
}
