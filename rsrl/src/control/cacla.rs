//! Continuous actor-critic learning automata
use crate::{
    domains::Transition,
    fa::StateActionUpdate,
    policies::Policy,
    Function,
    Handler,
};

/// Continuous Actor-Critic Learning Automaton
pub struct CACLA<C, P> {
    pub critic: C,
    pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
}

impl<C, P> CACLA<C, P> {
    pub fn new(critic: C, policy: P, alpha: f64, gamma: f64) -> Self {
        CACLA {
            critic,
            policy,

            alpha,
            gamma,
        }
    }
}

impl<'m, S, C, P> Handler<&'m Transition<S, P::Action>> for CACLA<C, P>
where
    C: Function<(&'m S,), Output = f64>,
    P: Policy<&'m S> + Handler<StateActionUpdate<&'m S, &'m <P as Policy<&'m S>>::Action, f64>>,

    P::Action: std::ops::Mul<f64, Output = f64>,
    &'m P::Action: std::ops::Sub<P::Action, Output = P::Action>,
{
    type Response = Option<P::Response>;
    type Error = P::Error;

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<Self::Response, Self::Error> {
        let s = t.from.state();
        let v = self.critic.evaluate((s,));
        let target = if t.terminated() {
            t.reward
        } else {
            t.reward + self.gamma * self.critic.evaluate((t.to.state(),))
        };

        if target > v {
            let mode = self.policy.mode(s);

            self.policy
                .handle(StateActionUpdate {
                    state: s,
                    action: &t.action,
                    error: (&t.action - mode) * self.alpha,
                })
                .map(|r| Some(r))
        } else {
            Ok(None)
        }
    }
}
