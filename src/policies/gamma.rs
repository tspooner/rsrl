extern crate special_fun;

use crate::{
    core::{Algorithm, Parameter},
    fa::{Parameterised, StateFunction, DifferentiableStateFunction},
    geometry::{Vector, Matrix, MatrixView, MatrixViewMut},
    policies::{DifferentiablePolicy, Policy},
};
use ndarray::Axis;
use rand::Rng;
use rstat::{
    Distribution, ContinuousDistribution,
    core::Modes,
    univariate::{UnivariateMoments, continuous::Gamma as GammaDist},
};
use serde::de::{self, Deserialize, Deserializer, Visitor, SeqAccess, MapAccess};
use std::{fmt, ops::AddAssign, marker::PhantomData};

const MIN_TOL: f64 = 0.1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gamma<A, T = A> {
    pub alpha: A,
    pub theta: T,
}

impl<A, T> Gamma<A, T> {
    pub fn new(alpha: A, theta: T) -> Self {
        Gamma { alpha, theta, }
    }

    #[inline]
    pub fn compute_alpha<S>(&self, s: &S) -> f64
        where A: StateFunction<S, Output = f64>,
    {
        self.alpha.evaluate(s).max(MIN_TOL)
    }

    #[inline]
    pub fn compute_theta<S>(&self, s: &S) -> f64
        where T: StateFunction<S, Output = f64>,
    {
        self.theta.evaluate(s).max(MIN_TOL)
    }

    #[inline]
    fn dist<S>(&self, input: &S) -> GammaDist
    where
        A: StateFunction<S, Output = f64>,
        T: StateFunction<S, Output = f64>,
    {
        GammaDist::new(self.compute_alpha(input), self.compute_theta(input))
    }

    fn gl_partial(&self, alpha: f64, theta: f64, x: f64) -> [f64; 2] {
        use special_fun::FloatSpecial;

        [(x / theta + 1e-5).ln() - alpha.digamma(), x / theta / theta - alpha / theta]
    }
}

impl<A, T> Algorithm for Gamma<A, T> {}

impl<S, A, T> Policy<S> for Gamma<A, T>
where
    A: StateFunction<S, Output = f64>,
    T: StateFunction<S, Output = f64>,
{
    type Action = f64;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, input: &S) -> f64 {
        self.dist(input).sample(rng)
    }

    fn mpa(&self, input: &S) -> f64 {
        self.dist(input).mean()
    }

    fn probability(&self, input: &S, a: &f64) -> f64 {
        self.dist(input).pdf(*a)
    }
}

impl<S, A, T> DifferentiablePolicy<S> for Gamma<A, T>
where
    A: DifferentiableStateFunction<S, Output = f64> + Parameterised,
    T: DifferentiableStateFunction<S, Output = f64> + Parameterised,
{
    fn update(&mut self, state: &S, a: &f64, error: f64) {
        let val_alpha = self.compute_alpha(state);
        let val_theta = self.compute_theta(state);

        let [gl_alpha, gl_theta] = self.gl_partial(val_alpha, val_theta, *a);

        self.alpha.update(state, gl_alpha * error);
        self.theta.update(state, gl_theta * error);
    }

    fn update_grad(&mut self, grad: &MatrixView<f64>) {
        match self.alpha.weights_dim() {
            [r, _] if r > 0 => {
                let grad_k = grad.slice(s![0..r, ..]);
                self.alpha.weights_view_mut().add_assign(&grad_k);
            },
            _ => {},
        }

        match self.theta.weights_dim() {
            [r, _] if r > 0 => {
                let grad_theta = grad.slice(s![self.theta.weights_dim()[0].., ..]);
                self.theta.weights_view_mut().add_assign(&grad_theta);
            },
            _ => {},
        }
    }

    fn update_grad_scaled(&mut self, grad: &MatrixView<f64>, factor: f64) {
        match self.alpha.weights_dim() {
            [r, _] if r > 0 => {
                let grad_k = grad.slice(s![0..self.alpha.weights_dim()[0], ..]);
                self.alpha.weights_view_mut().scaled_add(factor, &grad_k);
            },
            _ => {},
        }

        match self.theta.weights_dim() {
            [r, _] if r > 0 => {
                let grad_theta = grad.slice(s![self.theta.weights_dim()[0].., ..]);
                self.theta.weights_view_mut().scaled_add(factor, &grad_theta);
            },
            _ => {},
        }
    }

    fn grad_log(&self, state: &S, a: &f64) -> Matrix<f64> {
        let val_alpha = self.compute_theta(state);
        let grad_alpha: Matrix<f64> = self.alpha.grad(state).into();

        let val_theta = self.compute_theta(state);
        let grad_theta: Matrix<f64> = self.theta.grad(state).into();

        let [gl_alpha, gl_theta] = self.gl_partial(val_alpha, val_theta, *a);

        stack![Axis(0), gl_alpha * grad_alpha, gl_theta * grad_theta]
    }
}

impl<A: Parameterised, T: Parameterised> Parameterised for Gamma<A, T> {
    fn weights(&self) -> Matrix<f64> {
        stack![Axis(0), self.alpha.weights(), self.theta.weights()]
    }

    fn weights_view(&self) -> MatrixView<f64> {
        unimplemented!()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }

    fn weights_dim(&self) -> [usize; 2] {
        let [ra, _] = self.alpha.weights_dim();
        let [rb, _] = self.theta.weights_dim();

        [ra + rb, 1]
    }
}
