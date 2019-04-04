use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Parameterised, QFunction};
use crate::geometry::{MatrixView, MatrixViewMut};
use crate::policies::{Policy, FinitePolicy};
use std::marker::PhantomData;

/// On-policy variant of Watkins' Q-learning (aka "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
pub struct SARSA<Q, P> {
    pub q_func: Shared<Q>,
    pub policy: Shared<P>,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<Q, P> SARSA<Q, P> {
    pub fn new<T1, T2>(q_func: Shared<Q>, policy: Shared<P>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        SARSA {
            q_func,
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<Q, P: Algorithm> Algorithm for SARSA<Q, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.borrow_mut().handle_terminal();
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for SARSA<Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let qsa = self.q_func.evaluate_index(&self.q_func.to_features(s), t.action).unwrap();
        let residual = if t.terminated() {
            t.reward - qsa
        } else {
            let ns = t.to.state();
            let na = self.policy.borrow_mut().sample(ns);
            let nqsna = self.q_func.evaluate_index(&self.q_func.to_features(ns), na).unwrap();

            t.reward + self.gamma * nqsna - qsa
        };

        self.q_func.borrow_mut().update_index(
            &self.q_func.to_features(s),
            t.action, self.alpha * residual
        ).ok();
    }
}

impl<S, Q, P: Policy<S>> Controller<S, P::Action> for SARSA<Q, P> {
    fn sample_target(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }

    fn sample_behaviour(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }
}

impl<S, Q, P> ValuePredictor<S> for SARSA<Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.policy.borrow_mut().probabilities(s))
    }
}

impl<S, Q, P> ActionValuePredictor<S, P::Action> for SARSA<Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.evaluate(&self.q_func.to_features(s)).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.q_func.evaluate_index(&self.q_func.to_features(s), a).unwrap()
    }
}

impl<Q: Parameterised, P> Parameterised for SARSA<Q, P> {
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
