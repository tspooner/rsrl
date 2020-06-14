use crate::{
    domains::Batch,
    fa::StateActionUpdate,
    policies::Policy,
    Function,
    Handler,
};

#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct BaselineREINFORCE<B, P> {
    #[weights]
    pub policy: P,
    pub baseline: B,

    pub alpha: f64,
    pub gamma: f64,
}

impl<B, P> BaselineREINFORCE<B, P> {
    pub fn new(policy: P, baseline: B, alpha: f64, gamma: f64) -> Self {
        BaselineREINFORCE {
            policy,
            baseline,

            alpha,
            gamma,
        }
    }
}

impl<'m, S, B, P> Handler<&'m Batch<S, P::Action>> for BaselineREINFORCE<B, P>
where
    P: Policy<S> + Handler<StateActionUpdate<&'m S, &'m <P as Policy<S>>::Action>>,
    B: Function<(&'m S, &'m P::Action), Output = f64>,
{
    type Response = Vec<P::Response>;
    type Error = P::Error;

    fn handle(&mut self, batch: &'m Batch<S, P::Action>) -> Result<Self::Response, Self::Error> {
        let mut ret = 0.0;

        batch.iter().map(|t| {
            let s = t.from.state();
            let baseline = self.baseline.evaluate((s, &t.action));

            ret = t.reward + self.gamma * ret;

            self.policy.handle(StateActionUpdate {
                state: t.from.state(),
                action: &t.action,
                error: self.alpha * (ret - baseline),
            })
        }).collect()
    }
}
