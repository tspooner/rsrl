use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, Parameterised, Features, Projector, QFunction};
use crate::geometry::{MatrixView, MatrixViewMut};
use crate::policies::{fixed::Greedy, Policy};

/// True online variant of the Q(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
pub struct TOQLambda<F, P> {
    pub q_func: Shared<F>,

    pub policy: Shared<P>,
    pub target: Greedy<F>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    trace: Trace,
    q_old: f64,
}

impl<F, P> TOQLambda<F, P> {
    pub fn new<T1, T2>(
        trace: Trace,
        q_func: Shared<F>,
        policy: Shared<P>,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TOQLambda {
            q_func: q_func.clone(),

            policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),

            trace,
            q_old: 0.0,
        }
    }

    #[inline(always)]
    fn update_traces(&mut self, phi: Vector<f64>, decay_rate: f64) {
        let update_rate = self.trace.lambda.value() * self.gamma.value();
        let trace_update = (
            1.0 -
            self.alpha.value() * update_rate * self.trace.get().dot(&phi)
        ) * phi;

        self.trace.decay(decay_rate*update_rate);
        self.trace.update(&trace_update);
    }
}

impl<F, P> Algorithm for TOQLambda<F, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, F, P> OnlineLearner<S, P::Action> for TOQLambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = <Greedy<F> as Policy<S>>::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let phi_s = self.q_func.to_features(s);

        // Update traces:
        let decay_rate = if t.action == self.sample_target(s) { 1.0 } else { 0.0 };
        self.update_traces(
            phi_s.clone().expanded(self.q_func.n_features()),
            decay_rate,
        );

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
            let phi_ns = self.q_func.to_features(&ns);

            let na = self.sample_behaviour(ns);
            let nqsna = self.q_func.evaluate_index(&phi_ns, na).unwrap();

            self.q_old = nqsna;

            t.reward + self.gamma * nqsna - q_old
        };

        self.q_func.borrow_mut().update_index(
            &Features::Dense(z), t.action,
            self.alpha * residual,
        ).ok();

        self.q_func.borrow_mut().update_index(
            &phi_s, t.action,
            self.alpha * (q_old - qsa),
        ).ok();
    }
}

impl<S, F, P> Controller<S, P::Action> for TOQLambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = <Greedy<F> as Policy<S>>::Action>,
{
    fn sample_target(&mut self, s: &S) -> P::Action {
        self.target.sample(s)
    }

    fn sample_behaviour(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }
}

impl<S, F, P> ValuePredictor<S> for TOQLambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = <Greedy<F> as Policy<S>>::Action>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        let a = self.sample_target(s);

        self.predict_qsa(s, a)
    }
}

impl<S, F, P> ActionValuePredictor<S, P::Action> for TOQLambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = <Greedy<F> as Policy<S>>::Action>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.evaluate(&self.q_func.to_features(s)).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.q_func.evaluate_index(&self.q_func.to_features(s), a).unwrap()
    }
}

impl<F: Parameterised, P> Parameterised for TOQLambda<F, P> {
    fn weights(&self) -> Matrix<f64> {
        self.q_func.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.q_func.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }
}
