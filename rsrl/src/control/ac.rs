//! Actor-critic algorithms.
use crate::{
    domains::Transition,
    fa::StateActionUpdate,
    policies::Policy,
    Function,
    Handler,
};

pub trait Critic<S, A> {
    fn target(&self, t: &Transition<S, A>) -> f64;
}

impl<Q, S, A> Critic<S, A> for Q
where Q: for<'s> Function<(&'s S, &'s A), Output = f64>,
{
    fn target(&self, t: &Transition<S, A>) -> f64 {
        self.evaluate((t.from.state(), &t.action))
    }
}

pub struct TDCritic<V> {
    pub gamma: f64,
    pub value_function: V,
}

impl<V, S, A> Critic<S, A> for TDCritic<V>
where V: for<'s> Function<(&'s S,), Output = f64>,
{
    fn target(&self, t: &Transition<S, A>) -> f64 {
        let nv = self.value_function.evaluate((t.to.state(),));

        if t.terminated() {
            t.reward - nv
        } else {
            let v = self.value_function.evaluate((t.from.state(),));

            t.reward + self.gamma * nv - v
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct ActorCritic<C, P> {
    pub critic: C,
    pub policy: P,

    pub alpha: f64,
}

impl<C, P> ActorCritic<C, P> {
    pub fn new(critic: C, policy: P, alpha: f64) -> Self {
        ActorCritic {
            critic,
            policy,

            alpha,
        }
    }
}

impl<'m, S, C, P> Handler<&'m Transition<S, P::Action>> for ActorCritic<C, P>
where
    C: Critic<S, P::Action>,
    P: Policy<&'m S> + Handler<StateActionUpdate<&'m S, &'m <P as Policy<&'m S>>::Action, f64>>,
{
    type Response = P::Response;
    type Error = P::Error;

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<Self::Response, Self::Error> {
        self.policy.handle(StateActionUpdate {
            state: t.from.state(),
            action: &t.action,
            error: self.alpha * self.critic.target(&t),
        })
    }
}
