extern crate special_fun;

use crate::{
    fa::{
        StateFunction, DifferentiableStateFunction,
        Parameterised, Weights, WeightsView, WeightsViewMut,
    },
    policies::{DifferentiablePolicy, Policy},
};
use ndarray::{Array2, ArrayView2, Axis};
use rand::Rng;
use rstat::{
    Distribution, ContinuousDistribution,
    core::Modes,
    univariate::{UnivariateMoments, continuous::Beta as BetaDist},
};
use std::ops::AddAssign;

const MIN_TOL: f64 = 1.0;

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Beta<A, B = A> {
    pub alpha: A,
    pub beta: B,
}

impl<A, B> Beta<A, B> {
    pub fn new(alpha: A, beta: B) -> Self {
        Beta {
            alpha,
            beta,
        }
    }

    #[inline]
    pub fn compute_alpha<S>(&self, s: &S) -> f64 where A: StateFunction<S, Output = f64> {
        self.alpha.evaluate(s) + MIN_TOL
    }

    #[inline]
    pub fn compute_beta<S>(&self, s: &S) -> f64 where B: StateFunction<S, Output = f64> {
        self.beta.evaluate(s) + MIN_TOL
    }

    #[inline]
    fn dist<S>(&self, input: &S) -> BetaDist
    where
        A: StateFunction<S, Output = f64>,
        B: StateFunction<S, Output = f64>,
    {
        BetaDist::new(self.compute_alpha(input), self.compute_beta(input))
    }

    fn gl_partial(&self, alpha: f64, beta: f64, a: f64) -> [f64; 2] {
        use special_fun::FloatSpecial;

        const JITTER: f64 = 1e-9;

        let apb_digamma = (alpha + beta).digamma();
        let alpha_digamma = alpha.digamma();
        let beta_digamma = beta.digamma();

        [
            (a + JITTER).ln() - alpha_digamma + apb_digamma,
            (1.0 - a + JITTER).ln() - beta_digamma + apb_digamma
        ]
    }
}

impl<S, A, B> Policy<S> for Beta<A, B>
where
    A: StateFunction<S, Output = f64>,
    B: StateFunction<S, Output = f64>,
{
    type Action = f64;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, input: &S) -> f64 {
        self.dist(input).sample(rng)
    }

    fn mpa(&self, input: &S) -> f64 {
        let d = self.dist(input);
        let modes = d.modes();

        if modes.len() == 0 { d.mean() } else { modes[0] }
    }

    fn probability(&self, input: &S, a: &f64) -> f64 {
        self.dist(input).pdf(*a)
    }
}

impl<A: Parameterised, B: Parameterised> Parameterised for Beta<A, B> {
    fn weights(&self) -> Weights {
        stack![Axis(0), self.alpha.weights(), self.beta.weights()]
    }

    fn weights_view(&self) -> WeightsView {
        unimplemented!()
    }

    fn weights_view_mut(&mut self) -> WeightsViewMut {
        unimplemented!()
    }

    fn weights_dim(&self) -> [usize; 2] {
        let [ra, _] = self.alpha.weights_dim();
        let [rb, _] = self.beta.weights_dim();

        [ra + rb, 1]
    }
}

impl<S, A, B> DifferentiablePolicy<S> for Beta<A, B>
where
    A: DifferentiableStateFunction<S, Output = f64> + Parameterised,
    B: DifferentiableStateFunction<S, Output = f64> + Parameterised,
{
    fn update(&mut self, state: &S, a: &f64, error: f64) {
        let val_alpha = self.alpha.evaluate(state) + MIN_TOL;
        let val_beta = self.beta.evaluate(state) + MIN_TOL;

        let [gl_alpha, gl_beta] = self.gl_partial(val_alpha, val_beta, *a);

        self.alpha.update(state, gl_alpha * error);
        self.beta.update(state, gl_beta * error);
    }

    fn update_grad(&mut self, grad: &ArrayView2<f64>) {
        match self.alpha.weights_dim() {
            [r, _] if r > 0 => {
                let grad_alpha = grad.slice(s![0..r, ..]);
                self.alpha.weights_view_mut().add_assign(&grad_alpha);
            },
            _ => {},
        }

        match self.beta.weights_dim() {
            [r, _] if r > 0 => {
                let grad_beta = grad.slice(s![self.beta.weights_dim()[0].., ..]);
                self.beta.weights_view_mut().add_assign(&grad_beta);
            },
            _ => {},
        }
    }

    fn update_grad_scaled(&mut self, grad: &ArrayView2<f64>, factor: f64) {
        match self.alpha.weights_dim() {
            [r, _] if r > 0 => {
                let grad_alpha = grad.slice(s![0..self.alpha.weights_dim()[0], ..]);
                self.alpha.weights_view_mut().scaled_add(factor, &grad_alpha);
            },
            _ => {},
        }

        match self.beta.weights_dim() {
            [r, _] if r > 0 => {
                let grad_beta = grad.slice(s![self.beta.weights_dim()[0].., ..]);
                self.beta.weights_view_mut().scaled_add(factor, &grad_beta);
            },
            _ => {},
        }
    }

    fn grad_log(&self, state: &S, a: &f64) -> Array2<f64> {
        let val_alpha = self.alpha.evaluate(state) + MIN_TOL;
        let grad_alpha: Array2<f64> = self.alpha.grad(state).into();

        let val_beta = self.beta.evaluate(state) + MIN_TOL;
        let grad_beta: Array2<f64> = self.beta.grad(state).into();

        let [gl_alpha, gl_beta] = self.gl_partial(val_alpha, val_beta, *a);

        stack![Axis(0), gl_alpha * grad_alpha, gl_beta * grad_beta]
    }
}
