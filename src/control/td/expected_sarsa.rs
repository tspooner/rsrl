use core::{Algorithm, Predictor, Controller, Shared, Parameter, Vector, Matrix};
use domains::Transition;
use fa::{Parameterised, QFunction};
use policies::{Policy, FinitePolicy};
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
pub struct ExpectedSARSA<S, Q: QFunction<S>, P: Policy<S>> {
    pub q_func: Shared<Q>,
    pub policy: Shared<P>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q: QFunction<S>, P: Policy<S>> ExpectedSARSA<S, Q, P> {
    pub fn new<T1, T2>(q_func: Shared<Q>, policy: Shared<P>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        ExpectedSARSA {
            q_func,
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, Q: QFunction<S>, P: FinitePolicy<S>> ExpectedSARSA<S, Q, P> {
    #[inline(always)]
    fn update_q(&mut self, state: &S, action: P::Action, error: f64) {
        self.q_func.borrow_mut().update_action(state, action, self.alpha * error);
    }
}

impl<S, Q: QFunction<S>, P: FinitePolicy<S>> Algorithm<S, P::Action> for ExpectedSARSA<S, Q, P> {
    fn handle_sample(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qsa = self.predict_qsa(s, t.action);
        let exp_nv = self.predict_v(ns);
        let td_error = t.reward + self.gamma * exp_nv - qsa;

        self.update_q(&s, t.action, td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, P::Action>) {
        {
            let s = t.from.state();
            let qsa = self.predict_qsa(s, t.action);

            self.update_q(&s, t.action, t.reward - qsa);
        }

        self.policy.borrow_mut().handle_terminal(t);

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, Q: QFunction<S>, P: FinitePolicy<S>> Controller<S, P::Action> for ExpectedSARSA<S, Q, P> {
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, Q: QFunction<S>, P: FinitePolicy<S>> Predictor<S, P::Action> for ExpectedSARSA<S, Q, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.policy.borrow_mut().probabilities(s))
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}

impl<S, Q, P> Parameterised for ExpectedSARSA<S, Q, P>
where
    Q: QFunction<S> + Parameterised,
    P: FinitePolicy<S>,
{
    fn weights(&self) -> Matrix<f64> {
        self.q_func.borrow().weights()
    }
}
