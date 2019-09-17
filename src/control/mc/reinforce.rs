use crate::{
    BatchLearner,
    control::Controller,
    domains::Transition,
    fa::{Weights, WeightsView, WeightsViewMut, Parameterised},
    policies::{Policy, DifferentiablePolicy},
};
use rand::Rng;

#[derive(Clone, Debug, Serialize, Deserialize, Parameterised)]
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

impl<S, P> BatchLearner<S, P::Action> for REINFORCE<P>
where
    P: DifferentiablePolicy<S>,
    P::Action: Clone,
{
    fn handle_batch(&mut self, batch: &[Transition<S, P::Action>]) {
        let z = batch.len() as f64;
        let mut ret = 0.0;

        for t in batch.into_iter().rev() {
            ret = t.reward + self.gamma * ret;

            self.policy.update(
                t.from.state(),
                &t.action,
                self.alpha * ret / z
            );
        }
    }
}

impl<S, P: Policy<S>> Controller<S, P::Action> for REINFORCE<P> {
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}
