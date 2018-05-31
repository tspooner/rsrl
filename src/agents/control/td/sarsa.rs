use agents::{Controller, Predictor};
use domains::Transition;
use fa::QFunction;
use policies::Policy;
use std::marker::PhantomData;
use {Handler, Parameter, Vector};

/// On-policy variant of Watkins' Q-learning (aka "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
pub struct SARSA<S, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> SARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
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
    P: Policy,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s).unwrap();
        let nqs = self.q_func.evaluate(ns).unwrap();

        let a = t.action;
        let na = self.policy.sample(nqs.as_slice().unwrap());

        let td_error = t.reward + self.gamma * nqs[na] - qs[a];

        self.q_func.update_action(s, a, self.alpha * td_error);
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S, Q, P> Controller<S, usize> for SARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    fn pi(&mut self, s: &S) -> usize {
        self.policy
            .sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        p.sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }
}

impl<S, Q: QFunction<S>, P: Policy> Predictor<S> for SARSA<S, Q, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.q_func.evaluate(s).unwrap();
        let pi: Vector<f64> = self.policy.probabilities(nqs.as_slice().unwrap()).into();

        pi.dot(&nqs)
    }
}
