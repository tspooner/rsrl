use {Parameter};
use fa::{VFunction, QFunction};
use agents::{ControlAgent, PredictionAgent};
use domains::Transition;
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};
use std::marker::PhantomData;


/// Regular gradient descent actor critic.
pub struct ActorCritic<S: Space, P: Policy, Q, C>
    where Q: QFunction<S>,
          C: PredictionAgent<S>
{
    actor: Q,
    critic: C,

    policy: P,

    beta: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, P: Policy, Q, C> ActorCritic<S, P, Q, C>
    where Q: QFunction<S>,
          C: PredictionAgent<S>
{
    pub fn new<T1, T2>(actor: Q, critic: C, policy: P,
                           beta: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        ActorCritic {
            actor: actor,
            critic: critic,

            policy: policy,

            beta: beta.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, P: Policy, Q, C> ControlAgent<S, ActionSpace> for ActorCritic<S, P, Q, C>
    where Q: QFunction<S>,
          C: PredictionAgent<S>
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.actor.evaluate(s).as_slice())
    }

    fn pi_target(&mut self, s: &S::Repr) -> usize {
        Greedy.sample(self.actor.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let td_error = self.critic.handle_transition(s, ns, t.reward).unwrap();

        self.actor.update_action(s, t.action, self.beta*td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();
    }
}
