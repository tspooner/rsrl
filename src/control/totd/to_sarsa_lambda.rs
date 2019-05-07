use crate::{
    core::*,
    domains::Transition,
    fa::{Approximator, Parameterised, Features, QFunction},
    geometry::{MatrixView, MatrixViewMut},
    policies::{Policy, FinitePolicy},
};
use rand::{thread_rng, Rng};

/// True online variant of the SARSA(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
#[derive(Parameterised)]
pub struct TOSARSALambda<F, P> {
    #[weights] pub q_func: F,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    trace: Trace,
    q_old: f64,
}

impl<F, P> TOSARSALambda<F, P> {
    pub fn new<T1, T2>(
        q_func: F,
        policy: P,
        trace: Trace,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TOSARSALambda {
            q_func,
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            trace,
            q_old: 0.0,
        }
    }

    #[inline(always)]
    fn update_traces(&mut self, phi: Vector<f64>, decay_rate: f64) {
        let trace_update = (
            1.0 -
            self.alpha.value() * decay_rate * self.trace.get().dot(&phi)
        ) * phi;

        self.trace.decay(decay_rate);
        self.trace.update(&trace_update);
    }
}

impl<F, P> Algorithm for TOSARSALambda<F, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, F, P> OnlineLearner<S, P::Action> for TOSARSALambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = usize>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let phi_s = self.q_func.embed(s);

        // Update traces:
        let decay_rate = self.trace.lambda.value() * self.gamma.value();

        self.update_traces(phi_s.clone().expanded(self.q_func.n_features()), decay_rate);

        // Update weight vectors:
        let z = self.trace.get();
        let qsa = self.q_func.evaluate_index(&phi_s, t.action).unwrap();
        let q_old = self.q_old;

        let residual = if t.terminated() {
            self.q_old = 0.0;
            self.trace.decay(0.0);

            t.reward - q_old

        } else {
            let ns = t.to.state();
            let phi_ns = self.q_func.embed(ns);

            let na = self.sample_behaviour(&mut thread_rng(), ns);
            let nqsna = self.q_func.evaluate_index(&phi_ns, na).unwrap();

            self.q_old = nqsna;

            t.reward + self.gamma * nqsna - q_old
        };

        self.q_func.update_index(
            &Features::Dense(z),
            t.action,
            self.alpha * residual,
        ).ok();

        self.q_func.update_index(
            &phi_s, t.action,
            self.alpha * (q_old - qsa),
        ).ok();
    }
}

impl<S, F, P> Controller<S, P::Action> for TOSARSALambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S>,
{
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}

impl<S, F, P> ValuePredictor<S> for TOSARSALambda<F, P>
where
    F: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.policy.probabilities(s))
    }
}

impl<S, F, P> ActionValuePredictor<S, P::Action> for TOSARSALambda<F, P>
where
    F: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_qs(&self, s: &S) -> Vector<f64> {
        self.q_func.evaluate(&self.q_func.embed(s)).unwrap()
    }

    fn predict_qsa(&self, s: &S, a: P::Action) -> f64 {
        self.q_func.evaluate_index(&self.q_func.embed(s), a).unwrap()
    }
}
