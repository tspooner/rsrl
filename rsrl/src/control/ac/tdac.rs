use crate::{
    OnlineLearner,
    control::Controller,
    domains::Transition,
    policies::{Policy, DifferentiablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
};
use rand::Rng;

/// TD-error actor-critic.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TDAC<C, P> {
    pub critic: C,
    pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
}

impl<C, P> TDAC<C, P> {
    pub fn new(critic: C, policy: P, alpha: f64, gamma: f64) -> Self {
        TDAC {
            critic,
            policy,

            alpha,
            gamma,
        }
    }
}

impl<S, C, P> OnlineLearner<S, P::Action> for TDAC<C, P>
where
    C: OnlineLearner<S, P::Action> + ValuePredictor<S>,
    P: DifferentiablePolicy<S>,
    P::Action: Clone,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let v = self.critic.predict_v(s);
        let td_error = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * self.predict_v(t.to.state()) - v
        };

        self.critic.handle_transition(t);
        self.policy.update(s, &t.action, self.alpha * td_error);
    }

    fn handle_terminal(&mut self) {
        self.critic.handle_terminal();
    }
}

impl<S, C, P> ValuePredictor<S> for TDAC<C, P>
where
    C: ValuePredictor<S>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.critic.predict_v(s)
    }
}

impl<S, C, P> ActionValuePredictor<S, P::Action> for TDAC<C, P>
where
    C: ActionValuePredictor<S, P::Action>,
    P: Policy<S>,
{
    fn predict_q(&self, s: &S, a: &P::Action) -> f64 {
        self.critic.predict_q(s, a)
    }
}

impl<S, C, P> Controller<S, P::Action> for TDAC<C, P>
where
    P: Policy<S>,
{
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}
