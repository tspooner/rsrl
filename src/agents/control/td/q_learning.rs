use agents::{Controller, Predictor};
use domains::Transition;
use fa::QFunction;
use policies::{fixed::Greedy, Policy, FinitePolicy};
use std::marker::PhantomData;
use {Handler, Shared, Parameter};

/// Watkins' Q-learning.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLearning<S, Q: QFunction<S>, P: FinitePolicy<S>> {
    pub q_func: Shared<Q>,

    pub policy: P,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> QLearning<S, Q, P>
where
    Q: QFunction<S> + 'static,
    P: FinitePolicy<S>,
{
    pub fn new<T1, T2>(q_func: Shared<Q>, policy: P, alpha: T1, gamma: T2) -> Self
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

impl<S, Q, P> Handler<Transition<S, usize>> for QLearning<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
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

        self.policy.handle_terminal(t);
    }
}

impl<S, Q, P> Controller<S, usize> for QLearning<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn pi(&mut self, s: &S) -> usize { self.policy.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.target.sample(s) }
}

impl<S, Q: QFunction<S>, P: FinitePolicy<S>> Predictor<S> for QLearning<S, Q, P> {
    fn predict(&mut self, s: &S) -> f64 {
        self.q_func.borrow().evaluate(s).unwrap()[self.target.sample(s)]
    }
}
