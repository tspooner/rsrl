use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, Parameterised, Features, Projector, QFunction};
use crate::geometry::{MatrixView, MatrixViewMut};
use crate::policies::{Policy, FinitePolicy};

/// On-policy variant of Watkins' Q-learning with eligibility traces (aka
/// "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
pub struct SARSALambda<F, P> {
    pub fa_theta: Shared<F>,
    pub policy: Shared<P>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    trace: Trace,
}

impl<F, P> SARSALambda<F, P> {
    pub fn new<T1, T2>(
        fa_theta: Shared<F>,
        policy: Shared<P>,
        trace: Trace,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        SARSALambda {
            fa_theta,
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            trace,
        }
    }

    #[inline(always)]
    fn update_trace(&mut self, phi: Vector<f64>) {
        let decay_rate = self.trace.lambda.value() * self.gamma.value();

        self.trace.decay(decay_rate);
        self.trace.update(&phi);
    }
}

impl<F, P: Algorithm> Algorithm for SARSALambda<F, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.borrow_mut().handle_terminal();
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for SARSALambda<Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let phi_s = self.fa_theta.to_features(s);
        let qsa = self.fa_theta.evaluate_index(&phi_s, t.action).unwrap();

        // Update trace with latest feature vector:
        self.update_trace(phi_s.expanded(self.fa_theta.n_features()));

        // Update weight vectors:
        let z = self.trace.get();
        let residual = if t.terminated() {
            self.trace.decay(0.0);

            t.reward - qsa
        } else {
            let ns = t.to.state();
            let na = self.policy.borrow_mut().sample(ns);
            let nqsna = self.fa_theta.evaluate_index(&self.fa_theta.to_features(ns), na).unwrap();

            t.reward + self.gamma * nqsna - qsa
        };

        self.fa_theta.borrow_mut().update_index(
            &Features::Dense(z),
            t.action,
            self.alpha * residual,
        ).ok();
    }
}

impl<S, F, P: FinitePolicy<S>> Controller<S, P::Action> for SARSALambda<F, P> {
    fn sample_target(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }

    fn sample_behaviour(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }
}

impl<S, F, P> ValuePredictor<S> for SARSALambda<F, P>
where
    F: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.policy.borrow_mut().probabilities(s))
    }
}

impl<S, F, P> ActionValuePredictor<S, P::Action> for SARSALambda<F, P>
where
    F: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.fa_theta.evaluate(&self.fa_theta.to_features(s)).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.fa_theta.evaluate_index(&self.fa_theta.to_features(s), a).unwrap()
    }
}

impl<F: Parameterised, P> Parameterised for SARSALambda<F, P> {
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
