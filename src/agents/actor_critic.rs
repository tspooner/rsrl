use super::Agent;

use fa::{VFunction, QFunction};
use domains::Transition;
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

    alpha: f64,
    beta: f64,
    gamma: f64,

    phantom: PhantomData<S>,
}

impl<S: Space, P: Policy, Q, V> ActorCritic<S, P, Q, V>
    where Q: QFunction<S>,
          V: VFunction<S>
{
    pub fn new(actor: Q, critic: V, policy: P,
               alpha: f64, beta: f64, gamma: f64) -> Self
    {
        ActorCritic {
            actor: actor,
            critic: critic,

            policy: policy,

            alpha: alpha,
            beta: beta,
            gamma: gamma,

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
