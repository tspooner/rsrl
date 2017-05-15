use {Parameter};
use fa::{VFunction, QFunction};
use agents::{Agent, ControlAgent, PredictionAgent};
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

    alpha: Parameter,
    beta: Parameter,
    gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, P: Policy, Q, C> ActorCritic<S, P, Q, C>
    where Q: QFunction<S>,
          C: PredictionAgent<S>
{
    pub fn new<T1, T2, T3>(actor: Q, critic: C, policy: P,
                           alpha: T1, beta: T2, gamma: T3) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>
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

impl<S: Space, P: Policy, Q, C> Agent<S> for ActorCritic<S, P, Q, C>
    where Q: QFunction<S>,
          C: PredictionAgent<S>
{
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();
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

        let td_error = self.critic.handle_transition(s, ns, t.reward);

        self.actor.update_action(s, t.action, self.beta*td_error);
    }
}
