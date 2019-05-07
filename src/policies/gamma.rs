extern crate special_fun;

use crate::{
    core::{Algorithm, Parameter},
    fa::{Approximator, ScalarApproximator, Embedding, Features, Parameterised, VFunction},
    geometry::{Vector, Matrix, MatrixView, MatrixViewMut},
    policies::{DifferentiablePolicy, ParameterisedPolicy, Policy},
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

const MIN_TOL: f64 = 0.05;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gamma<A, B = A> {
    pub alpha: A,
    pub beta: B,
}

impl<A, B> Gamma<A, B> {
    pub fn new(alpha: A, beta: B) -> Self {
        Gamma {
            alpha,
            beta,
        }
    }

    #[inline]
    pub fn compute_alpha<S>(&self, s: &S) -> f64
        where A: VFunction<S>,
    {
        self.alpha.evaluate(&self.alpha.embed(s)).unwrap() + MIN_TOL
    }

    #[inline]
    pub fn compute_beta<S>(&self, s: &S) -> f64
        where B: VFunction<S>,
    {
        self.beta.evaluate(&self.beta.embed(s)).unwrap() + MIN_TOL
    }

    #[inline]
    fn dist<S>(&self, input: &S) -> GammaDist
        where A: VFunction<S>, B: VFunction<S>,
    {
        GammaDist::new(self.compute_alpha(input), self.compute_beta(input))
    }

    fn gl_partial(&self, alpha: f64, beta: f64, a: f64) -> [f64; 2]
        where A: ScalarApproximator, B: ScalarApproximator,
    {
        use special_fun::FloatSpecial;

        const JITTER: f64 = 1e-5;

        [beta.ln() + (a + JITTER).ln() - alpha.digamma(), alpha / beta - a]
    }
}

impl<A, B> Algorithm for Gamma<A, B> {}

impl<S, A: VFunction<S>, B: VFunction<S>> Policy<S> for Gamma<A, B> {
    type Action = f64;

    fn sample(&self, rng: &mut impl Rng, input: &S) -> f64 {
        self.dist(input).sample(rng)
    }

    fn mpa(&self, input: &S) -> f64 {
        self.dist(input).mean()
    }

    fn probability(&self, input: &S, a: &f64) -> f64 {
        self.dist(input).pdf(*a)
    }
}

impl<S, A, B> DifferentiablePolicy<S> for Gamma<A, B>
where
    A: VFunction<S> + Parameterised,
    B: VFunction<S> + Parameterised,
{
    fn grad_log(&self, input: &S, a: &f64) -> Matrix<f64> {
        let phi_alpha = self.alpha.embed(input);
        let val_alpha = self.alpha.evaluate(&phi_alpha).unwrap() + MIN_TOL;
        let jac_alpha = self.alpha.jacobian(&phi_alpha);

        let phi_beta = self.beta.embed(input);
        let val_beta = self.beta.evaluate(&phi_beta).unwrap() + MIN_TOL;
        let jac_beta = self.beta.jacobian(&phi_beta);

        let [gl_alpha, gl_beta] = self.gl_partial(val_alpha, val_beta, *a);

        stack![Axis(0), gl_alpha * jac_alpha, gl_beta * jac_beta]
    }
}

impl<A: Parameterised, B: Parameterised> Parameterised for Gamma<A, B> {
    fn weights(&self) -> Matrix<f64> {
        stack![Axis(0), self.alpha.weights(), self.beta.weights()]
    }

    fn weights_view(&self) -> MatrixView<f64> {
        unimplemented!()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }

    fn weights_dim(&self) -> (usize, usize) {
        (self.alpha.weights_dim().0 + self.beta.weights_dim().0, 1)
    }
}

impl<S, A, B> ParameterisedPolicy<S> for Gamma<A, B>
where
    A: VFunction<S> + Parameterised,
    B: VFunction<S> + Parameterised,
{
    fn update(&mut self, input: &S, a: &f64, error: f64) {
        let phi_alpha = self.alpha.embed(input);
        let val_alpha = self.alpha.evaluate(&phi_alpha).unwrap() + MIN_TOL;

        let phi_beta = self.beta.embed(input);
        let val_beta = self.beta.evaluate(&phi_beta).unwrap() + MIN_TOL;

        let [gl_alpha, gl_beta] = self.gl_partial(val_alpha, val_beta, *a);

        self.alpha.update(&phi_alpha, gl_alpha * error).ok();
        self.beta.update(&phi_beta, gl_beta * error).ok();
    }
}
