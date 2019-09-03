extern crate special_fun;

use crate::{
    core::{Algorithm, Parameter},
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction, DifferentiableStateFunction,
    },
    geometry::{Vector, Matrix, MatrixView, MatrixViewMut},
    policies::{DifferentiablePolicy, Policy},
};
use rand::Rng;
use rstat::{
    Distribution, ContinuousDistribution,
    core::Modes,
    multivariate::{MultivariateMoments, continuous::Dirichlet as DirichletDist},
};
use serde::de::{self, Deserialize, Deserializer, Visitor, SeqAccess, MapAccess};
use special_fun::FloatSpecial;
use std::{fmt, ops::{AddAssign, MulAssign}, marker::PhantomData};

const MIN_TOL: f64 = 1.0;

#[derive(Clone, Debug, Parameterised, Serialize, Deserialize)]
pub struct Dirichlet<F> {
    pub alphas: F,
}

impl<F> Dirichlet<F> {
    pub fn new(alphas: F) -> Self {
        Dirichlet { alphas, }
    }

    #[inline]
    pub fn compute_alphas<S>(&self, s: &S) -> Vec<f64>
        where F: StateFunction<S, Output = Vec<f64>>,
    {
        self.alphas.evaluate(s).into_iter().map(|x| x.max(0.0) + MIN_TOL).collect()
    }

    #[inline]
    fn dist<S>(&self, input: &S) -> DirichletDist
        where F: StateFunction<S, Output = Vec<f64>>,
    {
        DirichletDist::new(self.compute_alphas(input))
    }

    fn gl_iter<'a>(action: &'a [f64], alphas: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
        let sum_alphas: f64 = alphas.iter().sum();
        let digamma_sum_alphas = sum_alphas.digamma();

        action.into_iter()
            .map(|x| x.ln())
            .zip(alphas.into_iter().map(|a| a.digamma()))
            .map(move |(l, d)| l - d + digamma_sum_alphas)
    }
}

impl<F> Algorithm for Dirichlet<F> {}

impl<S, F> Policy<S> for Dirichlet<F>
where
    F: StateFunction<S, Output = Vec<f64>>,
{
    type Action = Vec<f64>;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, input: &S) -> Vec<f64> {
        self.dist(input).sample(rng)
    }

    fn mpa(&self, input: &S) -> Vec<f64> {
        self.dist(input).mean().into_raw_vec()
    }

    fn probability(&self, input: &S, a: &Vec<f64>) -> f64 {
        self.dist(input).pdf(a.clone())
    }
}

impl<S, F> DifferentiablePolicy<S> for Dirichlet<F>
where
    F: DifferentiableStateFunction<S, Output = Vec<f64>> + Parameterised,
{
    fn update(&mut self, state: &S, action: &Vec<f64>, error: f64) {
        let alphas = self.compute_alphas(state);
        let update = Self::gl_iter(&action, &alphas).map(|x| x * error).collect();

        self.alphas.update(state, update);
    }

    fn update_grad(&mut self, grad: &MatrixView<f64>) {
        self.alphas.weights_view_mut().add_assign(grad);
    }

    fn update_grad_scaled(&mut self, grad: &MatrixView<f64>, factor: f64) {
        self.alphas.weights_view_mut().scaled_add(factor, grad);
    }

    fn grad_log(&self, state: &S, a: &Vec<f64>) -> Matrix<f64> {
        let alphas = self.compute_alphas(state);
        let sum_alphas: f64 = alphas.iter().sum();
        let digamma_sum_alphas = sum_alphas.digamma();

        let gl = a.into_iter()
            .map(|x| x.ln())
            .zip(alphas.into_iter().map(|a| a.digamma()))
            .map(|(l, d)| l - d + digamma_sum_alphas);

        let mut grad_alphas: Matrix<f64> = self.alphas.grad(state).into();

        for (mut c, gl) in grad_alphas.gencolumns_mut().into_iter().zip(gl) {
            c.mul_assign(gl);
        }

        grad_alphas
    }
}
