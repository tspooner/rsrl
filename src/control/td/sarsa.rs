use core::{Controller, Predictor, Handler, Shared, Parameter, Vector};
use domains::Transition;
use fa::QFunction;
use policies::FinitePolicy;
use std::marker::PhantomData;

/// On-policy variant of Watkins' Q-learning (aka "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
pub struct SARSA<S, Q: QFunction<S>, P: FinitePolicy<S>> {
    pub q_func: Shared<Q>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> SARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    pub fn new<T1, T2>(q_func: Shared<Q>, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        SARSA {
            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, Q, P> Handler<Transition<S, usize>> for SARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qa = self.q_func.borrow().evaluate_action(s, t.action);
        let nqa = self.q_func.borrow().evaluate_action(ns, self.policy.sample(ns));

        let td_error = t.reward + self.gamma * nqa - qa;

        self.q_func.borrow_mut().update_action(s, t.action, self.alpha * td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal(t);
    }
}

impl<S, Q, P> Controller<S, usize> for SARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn pi(&mut self, s: &S) -> usize { self.policy.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }
}

impl<S, Q: QFunction<S>, P: FinitePolicy<S>> Predictor<S, usize> for SARSA<S, Q, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.policy.probabilities(s))
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: usize) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}
