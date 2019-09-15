use crate::{
    core::*,
    domains::Transition,
    fa::{Weights, WeightsView, WeightsViewMut, Parameterised},
    policies::{Policy, DifferentiablePolicy},
};
use rand::Rng;

#[derive(Clone, Debug, Serialize, Deserialize, Parameterised)]
pub struct REINFORCE<P> {
    #[weights] pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<P> REINFORCE<P> {
    pub fn new<T1, T2>(policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        REINFORCE {
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<P: Algorithm> Algorithm for REINFORCE<P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
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
