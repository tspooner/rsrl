use super::Agent;

use {Parameter};
use fa::{VFunction, QFunction};
use domain::Transition;
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};
use std::marker::PhantomData;


/// Regular gradient descent actor critic.
pub struct ActorCritic<S: Space, P: Policy, Q, V>
    where Q: QFunction<S>,
          V: VFunction<S>
{
    actor: Q,
    critic: V,

    policy: P,

    alpha: Parameter<f64>,
    beta: Parameter<f64>,
    gamma: Parameter<f64>,

    phantom: PhantomData<S>,
}

impl<S: Space, P: Policy, Q, V> ActorCritic<S, P, Q, V>
    where Q: QFunction<S>,
          V: VFunction<S>
{
    pub fn new<T1, T2, T3>(actor: Q, critic: V, policy: P,
                           alpha: T1, beta: T2, gamma: T3) -> Self
        where T1: Into<Parameter<f64>>,
              T2: Into<Parameter<f64>>,
              T3: Into<Parameter<f64>>
    {
        ActorCritic {
            actor: actor,
            critic: critic,

            policy: policy,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P: Policy, Q, V> Agent<S> for ActorCritic<S, P, Q, V>
    where Q: QFunction<S>,
          V: VFunction<S>
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.actor.evaluate(s).as_slice())
    }

    fn pi_target(&mut self, s: &S::Repr) -> usize {
        Greedy.sample(self.actor.evaluate(s).as_slice())
    }

    fn learn_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let delta = t.reward +
            self.gamma*self.critic.evaluate(ns) - self.critic.evaluate(s);

        self.actor.update_action(s, t.action, self.beta*delta);
        self.critic.update(s, self.alpha*delta);
    }
}
