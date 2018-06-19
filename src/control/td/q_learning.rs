use core::{Algorithm, Controller, Predictor, Shared, Parameter, Vector};
use domains::Transition;
use fa::QFunction;
use policies::{fixed::Greedy, Policy};
use std::marker::PhantomData;

/// Watkins' Q-learning.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLearning<S, Q: QFunction<S>, P: Policy<S>> {
    pub q_func: Shared<Q>,

    pub policy: Shared<P>,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q: QFunction<S> + 'static, P: Policy<S>> QLearning<S, Q, P> {
    pub fn new<T1, T2>(q_func: Shared<Q>, policy: Shared<P>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        QLearning {
            q_func: q_func.clone(),

            policy: policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, Q: QFunction<S>, P: Policy<S, Action = usize>> Algorithm<S, usize> for QLearning<S, Q, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.borrow().evaluate(s).unwrap();
        let nqs = self.q_func.borrow().evaluate(ns).unwrap();

        let td_error = t.reward + self.gamma * nqs[self.target.sample(&ns)] - qs[t.action];

        self.q_func.borrow_mut().update_action(s, t.action, self.alpha * td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.borrow_mut().handle_terminal(t);
    }
}

impl<S, Q: QFunction<S>, P: Policy<S, Action = usize>> Controller<S, usize> for QLearning<S, Q, P> {
    fn pi(&mut self, s: &S) -> usize { self.target.sample(s) }
    fn mu(&mut self, s: &S) -> usize { self.policy.borrow_mut().sample(s) }
}

impl<S, Q: QFunction<S>, P: Policy<S, Action = usize>> Predictor<S, usize> for QLearning<S, Q, P> {
    fn v(&mut self, s: &S) -> f64 {
        let a = self.target.sample(&s);

        self.qsa(s, a)
    }

    fn qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn qsa(&mut self, s: &S, a: usize) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}
