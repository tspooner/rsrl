use super::Agent;

use fa::{VFunction, QFunction};
use domain::Transition;
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};
use std::marker::PhantomData;


/// Watkins' Q-learning.
///
/// C. J. C. H. Watkins and P. Dayan, “Q-learning,” Mach. Learn., vol. 8, no. 3–4, pp. 279–292,
/// 1992.
pub struct QLearning<S: Space, P: Policy, Q: QFunction<S>>
{
    q_func: Q,
    policy: P,

    alpha: f64,
    gamma: f64,

    phantom: PhantomData<S>,
}

impl<S: Space, P: Policy, Q: QFunction<S>> QLearning<S, P, Q>
{
    pub fn new(q_func: Q, policy: P,
               alpha: f64, gamma: f64) -> Self
    {
        QLearning {
            q_func: q_func,
            policy: policy,

            alpha: alpha,
            gamma: gamma,

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P: Policy, Q: QFunction<S>> Agent<S> for QLearning<S, P, Q>
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

        let error = self.alpha*(t.reward + self.gamma*nqs[na] - qs[a]);

        self.q_func.update_action(s, a, error);
    }
}


/// Online Q-learning.
pub struct SARSA<S: Space, P: Policy, Q: QFunction<S>>
{
    q_func: Q,
    policy: P,

    alpha: f64,
    gamma: f64,

    phantom: PhantomData<S>
}

impl<S: Space, P: Policy, Q: QFunction<S>> SARSA<S, P, Q>
{
    pub fn new(q_func: Q, policy: P,
               alpha: f64, gamma: f64) -> Self
    {
        SARSA {
            q_func: q_func,
            policy: policy,

            alpha: alpha,
            gamma: gamma,

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P: Policy, Q: QFunction<S>> Agent<S> for SARSA<S, P, Q>
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

        let error = self.alpha*(t.reward + self.gamma*nqs[na] - qs[a]);

        self.q_func.update_action(s, a, error);
    }
}


// pub struct GreedyGQ<Q, V, P> {
    // q_func: Q,
    // v_func: V,

    // policy: P,

    // gamma: f64,
    // alpha: f64,
    // beta: f64,
// }

// impl<Q, V, P> GreedyGQ<Q, V, P> {
    // pub fn new(q_func: Q, v_func: V, policy: P, gamma: f64, alpha: f64, beta: f64) -> Self
    // {
        // GreedyGQ {
            // q_func: q_func,
            // v_func: v_func,

            // policy: policy,

            // alpha: alpha,
            // gamma: gamma,
            // beta: beta,
        // }
    // }
// }

// impl<S: Space, Q, V, P: Policy> Agent<S> for GreedyGQ<Q, V, P>
    // where Q: QFunction<S>,
          // V: VFunction<S>
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
        // let na = Greedy.sample(self.q_func.evaluate_phi(&phi_s).as_slice());

        // let error = t.reward +
            // self.q_func.evaluate_action_phi(&(self.gamma*phi_ns - phi_s), a);
        // let td_estimate = self.v_func.evaluate(s);

        // panic!("{:?}", error);
    // }
// }
