use crate::{
    Handler,
    domains::Batch,
    fa::StateActionUpdate,
    policies::Policy,
};

#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct REINFORCE<P> {
    #[weights] pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
}

impl<P> REINFORCE<P> {
    pub fn new(policy: P, alpha: f64, gamma: f64) -> Self {
        REINFORCE {
            policy,

            alpha,
            gamma,
        }
    }
}

impl<'m, S, P> Handler<&'m Batch<S, P::Action>> for REINFORCE<P>
where
    P: Policy<S> + Handler<StateActionUpdate<&'m S, &'m <P as Policy<S>>::Action>>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, batch: &'m Batch<S, P::Action>) -> Result<(), ()> {
        let mut ret = 0.0;

        for t in batch.into_iter().rev() {
            ret = t.reward + self.gamma * ret;

            self.policy.handle(StateActionUpdate {
                state: t.from.state(),
                action: &t.action,
                error: self.alpha * ret
            }).ok();
        }

        Ok(())
    }
}
