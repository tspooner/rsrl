use Parameter;
use agents::{ControlAgent, PredictionAgent};
use domains::Transition;
use fa::QFunction;
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};
use std::marker::PhantomData;


/// Regular gradient descent actor critic.
pub struct ActorCritic<S: Space, Q, C, P>
    where Q: QFunction<S>,
          C: PredictionAgent<S>,
          P: Policy
{
    pub actor: Q,
    pub critic: C,

    pub policy: P,

    pub beta: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, Q, C, P> ActorCritic<S, Q, C, P>
    where Q: QFunction<S>,
          C: PredictionAgent<S>,
          P: Policy
{
    pub fn new<T1, T2>(actor: Q, critic: C, policy: P, beta: T1, gamma: T2) -> Self
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

impl<S: Space, Q, C, P> ControlAgent<S, ActionSpace> for ActorCritic<S, Q, C, P>
    where Q: QFunction<S>,
          C: PredictionAgent<S>,
          P: Policy
{
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.policy.sample(self.actor.evaluate(s).as_slice())
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize {
        p.sample(self.actor.evaluate(s).as_slice())
    }

    fn handle_transition(&mut self, t: &Transition<S, ActionSpace>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let td_error = self.critic.handle_transition(s, ns, t.reward).unwrap();

        self.actor.update_action(s, t.action, self.beta*td_error);
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}
