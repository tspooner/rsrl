use super::Agent;

use {Function, Parameterised};
use fa::Linear;
use domain::Transition;
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};


/// Watkins' Q-learning agent.
///
/// C. J. C. H. Watkins and P. Dayan, “Q-learning,” Mach. Learn., vol. 8, no. 3–4, pp. 279–292,
/// 1992.
pub struct QLearning<Q, P> {
    q_func: Q,
    policy: P,

    alpha: f64,
    gamma: f64,
}

impl<Q, P> QLearning<Q, P> {
    pub fn new(q_func: Q, policy: P, alpha: f64, gamma: f64) -> Self {
        QLearning {
            q_func: q_func,
            policy: policy,

            alpha: alpha,
            gamma: gamma,
        }
    }
}

impl<S: Space, Q, P: Policy> Agent<S> for QLearning<Q, P>
    where Q: Function<S::Repr, Vec<f64>> + Parameterised<S::Repr, [f64]>
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn pi_target(&mut self, s: &S::Repr) -> usize {
        Greedy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn learn_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s);
        let nqs = self.q_func.evaluate(ns);

        let a = t.action;
        let na = Greedy.sample(nqs.as_slice());

        let mut errors = vec![0.0; qs.len()];
        errors[a] = self.alpha*(t.reward + self.gamma*nqs[na] - qs[a]);

        self.q_func.update(s, errors.as_slice());
    }
}


/// Online Q-learning agent.
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

    fn pi_target(&mut self, s: &S::Repr) -> usize {
        Greedy.sample(self.q_func.evaluate(s).as_slice())
    }

    fn learn_transition(&mut self, t: &Transition<S, ActionSpace>) {
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


// pub struct GreedyGQ<Q, P> {
    // q_func: Q,
    // policy: P,

    // alpha: f64,
    // gamma: f64,
    // eta: f64,
// }

// impl<Q, P> GreedyGQ<Q, P> {
    // pub fn new(q_func: Q, policy: P, alpha: f64, gamma: f64) -> Self {
        // GreedyGQ {
            // q_func: q_func,
            // policy: policy,

            // alpha: alpha,
            // gamma: gamma,
        // }
    // }
// }

// impl<S: Space, Q, P: Policy> Agent<S> for GreedyGQ<Q, P>
    // where Q: Function<S::Repr, Vec<f64>> + Parameterised<S::Repr, [f64]> + Linear<S::Repr>
// {
    // fn pi(&mut self, s: &S::Repr) -> usize {
        // self.policy.sample(self.q_func.evaluate(s).as_slice())
    // }

    // fn pi_target(&mut self, s: &S::Repr) -> usize {
        // Greedy.sample(self.q_func.evaluate(s).as_slice())
    // }

    // fn learn_transition(&mut self, t: &Transition<S, ActionSpace>) {
        // let (s, ns) = (t.from.state(), t.to.state());

        // let phi_s = self.q_func.phi(s);
        // let phi_ns = self.q_func.phi(ns);

        // let a = t.action;
        // let na = Greedy.sample(self.q_func.mul(phi_s).as_slice());

        // let error = t.reward + self.gamma*nqs[na] - qs[a];

        // println!("{:?}", phi_s);
    // }
// }
