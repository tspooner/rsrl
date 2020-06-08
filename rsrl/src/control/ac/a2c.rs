use crate::{
    Handler,
    domains::Transition,
    fa::StateActionUpdate,
    policies::Policy,
    prediction::{ValuePredictor, ActionValuePredictor},
};

/// Advantage actor-critic.
pub struct A2C<C, P> {
    pub critic: C,
    pub policy: P,

    pub alpha: f64,
}

impl<C, P> A2C<C, P> {
    pub fn new(critic: C, policy: P, alpha: f64) -> Self {
        A2C {
            critic,
            policy,

            alpha,
        }
    }
}

impl<'m, S, C, P> Handler<&'m Transition<S, P::Action>> for A2C<C, P>
where
    C: ValuePredictor<&'m S> + ActionValuePredictor<&'m S, &'m P::Action>,
    P: Policy<&'m S> + Handler<StateActionUpdate<&'m S, &'m <P as Policy<&'m S>>::Action, f64>>,
{
    type Response = P::Response;
    type Error = P::Error;

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<Self::Response, Self::Error> {
        let s = t.from.state();
        let v = self.critic.predict_v(s);
        let qsa = self.critic.predict_q(s, &t.action);

        self.policy.handle(StateActionUpdate {
            state: s,
            action: &t.action,
            error: self.alpha * (qsa - v),
        })
    }
}
