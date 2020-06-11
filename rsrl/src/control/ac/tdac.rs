use crate::{
    domains::Transition,
    fa::StateActionUpdate,
    policies::Policy,
    prediction::ValuePredictor,
    Handler,
};

/// TD-error actor-critic.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct TDAC<C, P> {
    pub critic: C,
    pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
}

impl<C, P> TDAC<C, P> {
    pub fn new(critic: C, policy: P, alpha: f64, gamma: f64) -> Self {
        TDAC {
            critic,
            policy,

            alpha,
            gamma,
        }
    }
}

impl<'m, S, C, P> Handler<&'m Transition<S, P::Action>> for TDAC<C, P>
where
    C: ValuePredictor<&'m S>,
    P: Policy<&'m S> + Handler<StateActionUpdate<&'m S, &'m <P as Policy<&'m S>>::Action, f64>>,
{
    type Response = P::Response;
    type Error = P::Error;

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<Self::Response, Self::Error> {
        let s = t.from.state();
        let v = self.critic.predict_v(s);
        let td_error = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * self.critic.predict_v(t.to.state()) - v
        };

        self.policy.handle(StateActionUpdate {
            state: s,
            action: &t.action,
            error: self.alpha * td_error,
        })
    }
}
