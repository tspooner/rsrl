extern crate special_fun;

use crate::{
    core::{Algorithm, Parameter},
    fa::{Approximator, ScalarApproximator, Embedding, Features, Parameterised, VFunction},
    geometry::{Vector, Matrix, MatrixView, MatrixViewMut},
    policies::{DifferentiablePolicy, ParameterisedPolicy, Policy},
};
use ndarray::Axis;
use rand::{thread_rng, rngs::{ThreadRng}};
use rstat::{
    Distribution, ContinuousDistribution,
    core::Modes,
    univariate::{UnivariateMoments, continuous::Beta as BetaDist},
};
use std::ops::AddAssign;

const MIN_TOL: f64 = 1.0;

#[derive(Clone, Debug, Serialize)]
pub struct Beta<A, B = A> {
    pub alpha: A,
    pub beta: B,

    #[serde(skip_serializing)]
    rng: ThreadRng,
}

impl<A, B> Beta<A, B> {
    pub fn new(alpha: A, beta: B) -> Self {
        Beta {
            alpha, beta,

            rng: thread_rng(),
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
    fn dist<S>(&self, input: &S) -> BetaDist
        where A: VFunction<S>, B: VFunction<S>,
    {
        BetaDist::new(self.compute_alpha(input), self.compute_beta(input))
    }

    fn gl_partial(&self, alpha: f64, beta: f64, a: f64) -> [f64; 2]
        where A: ScalarApproximator, B: ScalarApproximator,
    {
        use special_fun::FloatSpecial;

        const JITTER: f64 = 1e-5;

        let apb_digamma = (alpha + beta).digamma();
        let alpha_digamma = alpha.digamma();
        let beta_digamma = beta.digamma();

        [
            (a + JITTER).ln() - alpha_digamma + apb_digamma,
            (1.0 - a + JITTER).ln() - beta_digamma + apb_digamma
        ]
    }
}

impl<A, B> Algorithm for Beta<A, B> {}

impl<S, A: VFunction<S>, B: VFunction<S>> Policy<S> for Beta<A, B> {
    type Action = f64;

    fn sample(&mut self, input: &S) -> f64 {
        self.dist(input).sample(&mut self.rng)
    }

    fn mpa(&mut self, input: &S) -> f64 {
        let d = self.dist(input);
        let modes = d.modes();

        if modes.len() == 0 { d.mean() } else { modes[0] }
    }

    fn probability(&mut self, input: &S, a: f64) -> f64 {
        self.dist(input).pdf(a)
    }
}

impl<S, A: VFunction<S>, B: VFunction<S>> DifferentiablePolicy<S> for Beta<A, B> {
    fn grad_log(&self, input: &S, a: f64) -> Matrix<f64> {
        let phi_alpha = self.alpha.embed(input);
        let val_alpha = self.alpha.evaluate(&phi_alpha).unwrap() + MIN_TOL;
        let jac_alpha = self.alpha.jacobian(&phi_alpha);

        let phi_beta = self.beta.embed(input);
        let val_beta = self.beta.evaluate(&phi_beta).unwrap() + MIN_TOL;
        let jac_beta = self.beta.jacobian(&phi_beta);

        let [gl_alpha, gl_beta] = self.gl_partial(val_alpha, val_beta, a);

        stack![Axis(0), gl_alpha * jac_alpha, gl_beta * jac_beta]
    }
}

impl<F: Parameterised> Parameterised for Beta<F> {
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

impl<S, F: VFunction<S> + Parameterised> ParameterisedPolicy<S> for Beta<F> {
    fn update(&mut self, input: &S, a: f64, error: f64) {
        let phi_alpha = self.alpha.embed(input);
        let val_alpha = self.alpha.evaluate(&phi_alpha).unwrap() + MIN_TOL;

        let phi_beta = self.beta.embed(input);
        let val_beta = self.beta.evaluate(&phi_beta).unwrap() + MIN_TOL;

        let [gl_alpha, gl_beta] = self.gl_partial(val_alpha, val_beta, a);

        self.alpha.update(&phi_alpha, gl_alpha * error).ok();
        self.beta.update(&phi_beta, gl_beta * error).ok();
    }
}
