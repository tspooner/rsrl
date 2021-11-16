//! Actor-critic algorithms.
use crate::{domains::Transition, fa::StateActionUpdate, policies::Policy, Function, Handler};

pub trait Critic<'t, S: 't, A: 't> {
    fn target(&self, t: &'t Transition<S, A>) -> f64;
}

impl<'t, F, S: 't, A: 't> Critic<'t, S, A> for F
where
    F: Fn(&'t Transition<S, A>) -> f64,
{
    fn target(&self, t: &'t Transition<S, A>) -> f64 {
        (self)(t)
    }
}

pub struct QCritic<Q>(pub Q);

impl<'t, Q, S: 't, A: 't> Critic<'t, S, A> for QCritic<Q>
where
    Q: Function<(&'t S, &'t A), Output = f64>,
{
    fn target(&self, t: &'t Transition<S, A>) -> f64 {
        self.0.evaluate((t.from.state(), &t.action))
    }
}

pub struct TDCritic<V> {
    pub gamma: f64,
    pub v_func: V,
}

impl<'t, V, S: 't, A: 't> Critic<'t, S, A> for TDCritic<V>
where
    V: Function<(&'t S,), Output = f64>,
{
    fn target(&self, t: &'t Transition<S, A>) -> f64 {
        let nv = self.v_func.evaluate((t.to.state(),));

        if t.terminated() {
            t.reward - nv
        } else {
            let v = self.v_func.evaluate((t.from.state(),));

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

impl<Q, P> ActorCritic<QCritic<Q>, P> {
    pub fn qac(q_func: Q, policy: P, alpha: f64) -> Self {
        ActorCritic {
            critic: QCritic(q_func),
            policy,
            alpha,
        }
    }
}

impl<V, P> ActorCritic<TDCritic<V>, P> {
    pub fn tdac(v_func: V, policy: P, alpha: f64, gamma: f64) -> Self {
        ActorCritic {
            critic: TDCritic { gamma, v_func },
            policy,
            alpha,
        }
    }
}

impl<'m, S, C, P> Handler<&'m Transition<S, P::Action>> for ActorCritic<C, P>
where
    C: Critic<'m, S, P::Action>,
    P: Policy<&'m S> + Handler<StateActionUpdate<&'m S, &'m <P as Policy<&'m S>>::Action, f64>>,
{
    type Response = P::Response;
    type Error = P::Error;

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<Self::Response, Self::Error> {
        self.policy.handle(StateActionUpdate {
            state: t.from.state(),
            action: &t.action,
            error: self.alpha * self.critic.target(t),
        })
    }
}
