use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Parameterised, QFunction};
use crate::geometry::{MatrixView, MatrixViewMut};
use crate::policies::{Policy, FinitePolicy};
use std::marker::PhantomData;

/// Action probability-weighted variant of SARSA (aka "summation Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - van Seijen, H., van Hasselt, H., Whiteson, S., Wiering, M. (2009). A
/// theoretical and empirical analysis of Expected Sarsa. In Proceedings of the
/// IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning,
/// pp. 177–184.
pub struct ExpectedSARSA<Q, P> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<Q, P> ExpectedSARSA<Q, P> {
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        ExpectedSARSA {
            q_func,
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<Q, P: Algorithm> Algorithm for ExpectedSARSA<Q, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for ExpectedSARSA<Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let qsa = self.predict_qsa(s, t.action);
        let residual = if t.terminated() {
            t.reward - qsa
        } else {
            let ns = t.to.state();
            let exp_nv = self.predict_v(ns);

            t.reward + self.gamma * exp_nv - qsa
        };

        self.q_func.update_index(
            &self.q_func.to_features(s),
            t.action, self.alpha * residual
        ).ok();
    }
}

impl<S, Q, P: Policy<S>> Controller<S, P::Action> for ExpectedSARSA<Q, P> {
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.sample(s) }
}

impl<S, Q, P> ValuePredictor<S> for ExpectedSARSA<Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.policy.probabilities(s))
    }
}

impl<S, Q, P> ActionValuePredictor<S, P::Action> for ExpectedSARSA<Q, P>
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

impl<Q: Parameterised, P> Parameterised for ExpectedSARSA<Q, P> {
    fn weights(&self) -> Matrix<f64> {
        self.q_func.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.q_func.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.q_func.weights_view_mut()
    }
}
