use core::{Algorithm, Predictor, Controller, Shared, Parameter, Vector};
use domains::Transition;
use fa::QFunction;
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
            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, Q: QFunction<S>, P: FinitePolicy<S>> Algorithm<S, usize> for ExpectedSARSA<S, Q, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qa = self.q_func.borrow().evaluate_action(s, t.action);
        let exp_nv = self.v(ns);
        let td_error = t.reward + self.gamma * exp_nv - qa;

        self.q_func.borrow_mut().update_action(s, t.action, self.alpha * td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.borrow_mut().handle_terminal(t);
    }
}

impl<S, Q: QFunction<S>, P: FinitePolicy<S>> Controller<S, usize> for ExpectedSARSA<S, Q, P> {
    fn pi(&mut self, s: &S) -> usize { self.policy.borrow_mut().sample(s) }
    fn mu(&mut self, s: &S) -> usize { self.policy.borrow_mut().sample(s) }
}

impl<S, Q: QFunction<S>, P: FinitePolicy<S>> Predictor<S, usize> for ExpectedSARSA<S, Q, P> {
    fn v(&mut self, s: &S) -> f64 {
        self.qs(s).dot(&self.policy.borrow_mut().probabilities(s))
    }

    fn qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn qsa(&mut self, s: &S, a: usize) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}
