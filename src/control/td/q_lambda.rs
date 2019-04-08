use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, Parameterised, Features, Projector, QFunction};
use crate::geometry::{MatrixView, MatrixViewMut};
use crate::policies::{Greedy, Policy};

/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLambda<F, P> {
    pub fa_theta: F,

    pub policy: P,
    pub target: Greedy<F>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    trace: Trace,
}

impl<F, P> QLambda<Shared<F>, P> {
    pub fn new<T1, T2>(
        fa_theta: F,
        policy: P,
        trace: Trace,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        let fa_theta = make_shared(fa_theta);

        QLambda {
            fa_theta: fa_theta.clone(),

            policy,
            target: Greedy::new(fa_theta),

            alpha: alpha.into(),
            gamma: gamma.into(),

            trace,
        }
    }
}

impl<F, P: Algorithm> Algorithm for QLambda<F, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
        self.target.handle_terminal();
    }
}

impl<S, F, P> OnlineLearner<S, P::Action> for QLambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = <Greedy<F> as Policy<S>>::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let phi_s = self.fa_theta.to_features(s);
        let qsa = self.fa_theta.evaluate_index(&phi_s, t.action).unwrap();

        // Update trace:
        let decay_rate = if t.action == self.target.sample(s) {
            self.trace.lambda.value() * self.gamma.value()
        } else {
            0.0
        };

        self.trace.decay(decay_rate);
        self.trace.update(&phi_s.expanded(self.fa_theta.n_features()));

        // Update weight vectors:
        let z = self.trace.get();
        let residual = if t.terminated() {
            self.trace.decay(0.0);

            t.reward - qsa
        } else {
            let ns = t.to.state();
            let phi_ns = self.fa_theta.to_features(ns);

            let na = self.target.sample(&ns);
            let nqsna = self.fa_theta.evaluate_index(&phi_ns, na).unwrap();

            t.reward + self.gamma * nqsna - qsa
        };

        self.fa_theta.update_index(
            &Features::Dense(z),
            t.action,
            self.alpha * residual,
        ).ok();
    }
}

impl<S, F, P> Controller<S, P::Action> for QLambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = <Greedy<F> as Policy<S>>::Action>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.sample(s) }
}

impl<S, F, P> ValuePredictor<S> for QLambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = <Greedy<F> as Policy<S>>::Action>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        let a = self.target.sample(s);

        self.predict_qsa(s, a)
    }
}

impl<S, F, P> ActionValuePredictor<S, P::Action> for QLambda<F, P>
where
    F: QFunction<S>,
    P: Policy<S, Action = <Greedy<F> as Policy<S>>::Action>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.fa_theta.evaluate(&self.fa_theta.to_features(s)).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.fa_theta.evaluate_index(&self.fa_theta.to_features(s), a).unwrap()
    }
}

impl<F: Parameterised, P> Parameterised for QLambda<F, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.fa_theta.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }
}
