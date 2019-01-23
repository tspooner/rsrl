use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Parameterised, QFunction};
use crate::policies::{fixed::Greedy, Policy, FinitePolicy};
use std::marker::PhantomData;

/// Watkins' Q-learning.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLearning<Q, P> {
    pub q_func: Shared<Q>,

    pub policy: Shared<P>,
    pub target: Greedy<Q>,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<Q, P> QLearning<Q, P> {
    pub fn new<T1, T2>(q_func: Shared<Q>, policy: Shared<P>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        QLearning {
            q_func: q_func.clone(),

            policy,
            target: Greedy::new(q_func),

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<Q, P: Algorithm> Algorithm for QLearning<Q, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.borrow_mut().handle_terminal();
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for QLearning<Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qsa = self.predict_qsa(&s, t.action);
        let residual = if t.terminated() {
            t.reward - qsa
        } else {
            let na = self.sample_target(&ns);
            let nqsna = self.predict_qsa(&ns, na);

            t.reward + self.gamma * nqsna - qsa
        };

        self.q_func.borrow_mut().update_action(s, t.action, self.alpha * residual);
    }
}

impl<S, Q, P> Controller<S, P::Action> for QLearning<Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, Q, P> ValuePredictor<S> for QLearning<Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        let a = self.target.sample(s);

        self.predict_qsa(s, a)
    }
}

impl<S, Q, P> ActionValuePredictor<S, P::Action> for QLearning<Q, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}

impl<Q: Parameterised, P> Parameterised for QLearning<Q, P> {
    fn weights(&self) -> Matrix<f64> {
        self.q_func.borrow().weights()
    }
}
