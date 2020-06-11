use crate::{
    Handler, OutputOf,
    domains::Transition,
    fa::linear::basis::Basis,
    params::*,
    policies::{Policy, EnumerablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
    traces::Trace,
    utils::argmax_first,
};
use ndarray::{Array1, Ix1, linalg::Dot};
use std::f64;

/// True online variant of the Q(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
#[derive(Parameterised)]
pub struct TOQLambda<B, P, T> {
    pub basis: B,
    #[weights] pub theta: Array1<f64>,
    pub trace: T,

    pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
    pub lambda: f64,

    q_old: f64,
}

impl<B, P, T> TOQLambda<B, P, T> {
    pub fn new(
        basis: B,
        theta: Array1<f64>,
        trace: T,
        policy: P,
        alpha: f64,
        gamma: f64,
        lambda: f64,
    ) -> Self {
        TOQLambda {
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

        TOQLambda::new(
            basis, Array1::zeros(n), trace,
            policy, alpha, gamma, lambda,
        )
    }
}

impl<'m, S, B, P, T> Handler<&'m Transition<S, P::Action>> for TOQLambda<B, P, T>
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

        let phi_s: Vec<_> = (0..self.policy.len((s,)))
            .into_iter()
            .map(|a| self.basis.project((s, a)).unwrap())
            .collect();
        let phi_s_a = &phi_s[t.action];

        let qs: Vec<_> = phi_s.iter().map(|f| f.dot(&self.theta)).collect();
        let qsa = qs[t.action];

        let (amax, _) = argmax_first(qs);

        if t.action == amax {
            let a = self.alpha;
            let c = self.lambda * self.gamma;

            let dotted = self.trace.deref().dot(phi_s_a);

            self.trace.merge_inplace(&phi_s_a, move |x, y| {
                c * x + (1.0 - a * c * dotted) * y
            });
        } else {
            self.trace.merge_inplace(&phi_s_a, |_, y| y);
        }

        // Update weight vectors:
        if t.terminated() {
            self.trace.deref().scaled_addto(self.alpha * (t.reward - self.q_old), &mut self.theta);
            phi_s_a.scaled_addto(self.alpha * (self.q_old - qsa), &mut self.theta);

            self.q_old = 0.0;
            self.trace.reset();
        } else {
            let ns = t.to.state();
            let phi_ns_na = self.basis.project((ns, 0)).unwrap();
            let qnsna = phi_ns_na.dot(&self.theta);

            let (phi_ns_na, qnsna) = (1..self.policy.len((ns,)))
                .into_iter()
                .fold((phi_ns_na, qnsna), |acc, a| {
                    let phi = self.basis.project((s, a)).unwrap();
                    let val = phi.dot(&self.theta);

                    if val - acc.1 > 1e-7 { (phi, val) } else { acc }
                });

            let residual = t.reward + self.gamma * qnsna - self.q_old;

            self.trace.deref().scaled_addto(self.alpha * residual, &mut self.theta);
            phi_ns_na.scaled_addto(self.alpha * (self.q_old - qsa), &mut self.theta);

            self.q_old = qnsna;
            if t.action != amax {
                self.trace.reset();
            }
        }

        Ok(())
    }
}

impl<S, B, P, T> ValuePredictor<S> for TOQLambda<B, P, T>
where
    B: for<'s> Basis<(&'s S, usize)>,
    P: for<'s> EnumerablePolicy<&'s S>,

    B::Value: Dot<Array1<f64>, Output = f64>,

    for<'s> OutputOf<P, (&'s S,)>: std::ops::Index<usize, Output = f64> + IntoIterator<Item = f64>,
    for<'s> <OutputOf<P, (&'s S,)> as IntoIterator>::IntoIter: ExactSizeIterator,
{
    fn predict_v(&self, s: S) -> f64 {
        (0..self.policy.len((&s,)))
            .into_iter()
            .map(|a| self.basis.project((&s, a)).unwrap().dot(&self.theta))
            .fold(f64::MIN, |acc, x| if x - acc > 1e-7 { x } else { acc })
    }
}

impl<S, B, P, T> ActionValuePredictor<S, P::Action> for TOQLambda<B, P, T>
where
    B: Basis<(S, P::Action)>,
    P: Policy<S>,

    B::Value: Dot<Array1<f64>, Output = f64>,
{
    fn predict_q(&self, s: S, a: P::Action) -> f64 {
        self.basis.project((s, a)).unwrap().dot(&self.theta)
    }
}
