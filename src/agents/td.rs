use super::Agent;

use {Function, Parameterised};
use domain::Transition;
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};


pub struct QLearning<Q, P> {
    q_func: Q,

    behaviour_policy: P,
    greedy_policy: Greedy,

    alpha: f64,
    gamma: f64,
}

impl<Q, P> QLearning<Q, P> {
    pub fn new(q_func: Q, policy: P, alpha: f64, gamma: f64) -> Self {
        QLearning {
            q_func: q_func,

            behaviour_policy: policy,
            greedy_policy: Greedy,

            alpha: alpha,
            gamma: gamma,
        }
    }
}

impl<S: Space, Q, P: Policy> Agent<S> for QLearning<Q, P>
    where Q: Function<S::Repr, Vec<f64>> + Parameterised<S::Repr, [f64]>
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.behaviour_policy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn train(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s);
        let nqs = self.q_func.evaluate(ns);

        let a = t.action;
        let na = self.greedy_policy.sample(nqs.as_slice());

        let mut errors = vec![0.0; qs.len()];
        errors[a] = self.alpha*(t.reward + self.gamma*nqs[na] - qs[a]);

        self.q_func.update(s, errors.as_slice());
    }
}


pub struct SARSA<Q, P> {
    q_func: Q,
    policy: P,

    alpha: f64,
    gamma: f64,
}

impl<Q, P> SARSA<Q, P> {
    pub fn new(q_func: Q, policy: P, alpha: f64, gamma: f64) -> Self {
        SARSA {
            q_func: q_func,
            policy: policy,

            alpha: alpha,
            gamma: gamma,
        }
    }
}

impl<S: Space, Q, P: Policy> Agent<S> for SARSA<Q, P>
    where Q: Function<S::Repr, Vec<f64>> + Parameterised<S::Repr, [f64]>
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn train(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s);
        let nqs = self.q_func.evaluate(ns);

        let a = t.action;
        let na = self.policy.sample(nqs.as_slice());

        let mut errors = vec![0.0; qs.len()];
        errors[a] = self.alpha*(t.reward + self.gamma*nqs[na] - qs[a]);

        self.q_func.update(s, errors.as_slice());
    }
}
