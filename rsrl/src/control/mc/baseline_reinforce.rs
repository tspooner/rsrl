use crate::{
    BatchLearner,
    control::Controller,
    domains::Transition,
    fa::{Weights, WeightsView, WeightsViewMut, Parameterised},
    policies::{Policy, DifferentiablePolicy},
    prediction::ActionValuePredictor,
};
use rand::Rng;

#[derive(Parameterised)]
pub struct BaselineREINFORCE<B, P> {
    #[weights] pub policy: P,
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

impl<S, B, P> BatchLearner<S, P::Action> for BaselineREINFORCE<B, P>
where
    S: Clone,
    P: DifferentiablePolicy<S>,
    P::Action: Clone,
    B: ActionValuePredictor<S, P::Action>,
{
    fn handle_batch(&mut self, batch: &[Transition<S, P::Action>]) {
        let mut ret = 0.0;

        for t in batch.into_iter().rev() {
            let s = t.from.state();
            let baseline = self.baseline.predict_qsa(s, t.action.clone());

            ret = t.reward + self.gamma * ret;

            self.policy.update(s, &t.action, self.alpha * (ret - baseline));
        }
    }
}

impl<S, B, P: Policy<S>> Controller<S, P::Action> for BaselineREINFORCE<B, P> {
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}
