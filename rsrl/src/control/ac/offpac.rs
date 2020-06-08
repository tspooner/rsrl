use crate::{
    Handler,
    domains::Transition,
    fa::StateActionUpdate,
    policies::Policy,
    prediction::ValuePredictor,
};

/// Off-policy TD-based actor-critic.
pub struct OffPAC<C, P, B> {
    pub critic: C,
    pub target: P,
    pub behaviour: B,

    pub alpha: f64,
    pub gamma: f64,
}

impl<C, P, B> OffPAC<C, P, B> {
    pub fn new(
        critic: C,
        target: P,
        behaviour: B,
        alpha: f64,
        gamma: f64,
    ) -> Self {
        OffPAC {
            critic,
            target,
            behaviour,

            alpha,
            gamma,
        }
    }
}

impl<'m, S, C, P, B> Handler<&'m Transition<S, P::Action>> for OffPAC<C, P, B>
where
    C: ValuePredictor<&'m S>,
    B: Policy<&'m S>,
    P: Policy<&'m S, Action = B::Action> +
        Handler<StateActionUpdate<&'m S, &'m <P as Policy<&'m S>>::Action, f64>>,
{
    type Response = P::Response;
    type Error = P::Error;

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<Self::Response, Self::Error> {
        let (s, ns) = (t.from.state(), t.to.state());

        let v = self.critic.predict_v(s);

        let residual = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * self.critic.predict_v(ns) - v
        };

        let is_ratio = {
            let pi = self.target.evaluate((s, &t.action));
            let b = self.behaviour.evaluate((s, &t.action));

            pi / b
        };

        self.target.handle(StateActionUpdate {
            state: s,
            action: &t.action,
            error: self.alpha * residual * is_ratio,
        })
    }
}
