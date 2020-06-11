use crate::{
    domains::Transition,
    fa::StateActionUpdate,
    policies::Policy,
    prediction::ActionValuePredictor,
    Handler,
};

/// Action-value actor-critic.
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct QAC<C, P> {
    pub critic: C,
    pub policy: P,

    pub alpha: f64,
}

impl<C, P> QAC<C, P> {
    pub fn new(critic: C, policy: P, alpha: f64) -> Self {
        QAC {
            critic,
            policy,

            alpha,
        }
    }
}

impl<'m, S, C, P> Handler<&'m Transition<S, P::Action>> for QAC<C, P>
where
    C: ActionValuePredictor<&'m S, &'m P::Action>,
    P: Policy<&'m S> + Handler<StateActionUpdate<&'m S, &'m <P as Policy<&'m S>>::Action, f64>>,
{
    type Response = P::Response;
    type Error = P::Error;

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<Self::Response, Self::Error> {
        let s = t.from.state();
        let qsa = self.critic.predict_q(s, &t.action);

        self.policy.handle(StateActionUpdate {
            state: s,
            action: &t.action,
            error: self.alpha * qsa,
        })
    }
}
