use crate::{
    Handler, OutputOf,
    domains::Transition,
    fa::linear::basis::Basis,
    params::*,
    policies::{Policy, EnumerablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
    traces::Trace,
};
use ndarray::{Array1, Ix1, linalg::Dot};
use rand::thread_rng;
use std::f64;

/// True online variant of the SARSA(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
#[derive(Parameterised)]
pub struct TOSARSALambda<B, P, T> {
    pub basis: B,
    #[weights] pub theta: Array1<f64>,
    pub trace: T,

    pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
    pub lambda: f64,

    q_old: f64,
}

impl<B, P, T> TOSARSALambda<B, P, T> {
    pub fn new(
        basis: B,
        theta: Array1<f64>,
        trace: T,
        policy: P,
        alpha: f64,
        gamma: f64,
        lambda: f64,
    ) -> Self {
        TOSARSALambda {
            basis,
            theta,
            trace,

            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),
            lambda: lambda.into(),

            q_old: 0.0,
        }
    }

    pub fn zeros(
        basis: B,
        trace: T,
        policy: P,
        alpha: f64,
        gamma: f64,
        lambda: f64,
    ) -> Self
    where
        B: spaces::Space,
    {
        let n: usize = basis.dim().into();

        TOSARSALambda::new(
            basis, Array1::zeros(n), trace,
            policy, alpha, gamma, lambda,
        )
    }
}

impl<'m, S, B, P, T> Handler<&'m Transition<S, P::Action>> for TOSARSALambda<B, P, T>
where
    B: Basis<(&'m S, &'m P::Action)> + Basis<(&'m S, P::Action)>,
    P: EnumerablePolicy<&'m S>,
    T: Trace<Buffer = B::Value>,

    B::Value: BufferMut<Dim = Ix1> +
        Dot<Array1<f64>, Output = f64> +
        Dot<B::Value, Output = f64>,

    OutputOf<P, (&'m S,)>: std::ops::Index<usize, Output = f64> + IntoIterator<Item = f64>,
    <OutputOf<P, (&'m S,)> as IntoIterator>::IntoIter: ExactSizeIterator,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<(), ()> {
        let s = t.from.state();

        // Update trace with latest feature vector:
        let phi_s_a = self.basis.project((s, &t.action)).unwrap();

        let qsa = phi_s_a.dot(&self.theta);

        {
            let a = self.alpha;
            let c = self.lambda * self.gamma;

            let dotted = self.trace.deref().dot(&phi_s_a);

            self.trace.merge_inplace(&phi_s_a, move |x, y| {
                c * x + (1.0 - a * c * dotted) * y
            });
        }

        if t.terminated() {
            self.trace.deref().scaled_addto(self.alpha * (t.reward - self.q_old), &mut self.theta);
            phi_s_a.scaled_addto(self.alpha * (self.q_old - qsa), &mut self.theta);

            self.q_old = 0.0;
            self.trace.reset();
        } else {
            let ns = t.to.state();
            let na = self.policy.sample(&mut thread_rng(), ns);

            let phi_ns_na = self.basis.project((ns, na)).unwrap();
            let qnsna = phi_ns_na.dot(&self.theta);
            let residual = t.reward + self.gamma * qnsna - qsa;

            self.trace.deref().scaled_addto(self.alpha * residual, &mut self.theta);
            phi_ns_na.scaled_addto(self.alpha * (self.q_old - qsa), &mut self.theta);

            self.q_old = qnsna;
        };

        Ok(())
    }
}

impl<S, B, P, T> ValuePredictor<S> for TOSARSALambda<B, P, T>
where
    B: for<'s> Basis<(&'s S, usize)>,
    P: for<'s> EnumerablePolicy<&'s S>,

    B::Value: Dot<Array1<f64>, Output = f64>,

    for<'s> OutputOf<P, (&'s S,)>: std::ops::Index<usize, Output = f64> + IntoIterator<Item = f64>,
    for<'s> <OutputOf<P, (&'s S,)> as IntoIterator>::IntoIter: ExactSizeIterator,
{
    fn predict_v(&self, s: S) -> f64 {
        self.policy
            .evaluate((&s,))
            .into_iter()
            .enumerate()
            .fold(0.0, |acc, (a, p)| {
                acc + p * self.basis.project((&s, a)).unwrap().dot(&self.theta)
            })
    }
}

impl<S, B, P, T> ActionValuePredictor<S, P::Action> for TOSARSALambda<B, P, T>
where
    B: Basis<(S, P::Action)>,
    P: Policy<S>,

    B::Value: Dot<Array1<f64>, Output = f64>,
{
    fn predict_q(&self, s: S, a: P::Action) -> f64 {
        self.basis.project((s, a)).unwrap().dot(&self.theta)
    }
}
