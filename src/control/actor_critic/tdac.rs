use crate::core::*;
use crate::domains::Transition;
use crate::policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

/// TD-error actor-critic.
pub struct TDAC<C, P> {
    pub critic: C,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<C, P> TDAC<C, P> {
    pub fn new<T1, T2>(critic: C, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TDAC {
            critic,
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<C, P> Algorithm for TDAC<C, P>
where
    C: Algorithm,
    P: Algorithm,
{
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.critic.handle_terminal();
        self.policy.handle_terminal();
    }
}

impl<S, C, P> OnlineLearner<S, P::Action> for TDAC<C, P>
where
    C: OnlineLearner<S, P::Action> + ValuePredictor<S>,
    P: ParameterisedPolicy<S>,
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
        self.policy.update(s, t.action.clone(), self.alpha * td_error);
    }
}

impl<S, C, P> ValuePredictor<S> for TDAC<C, P>
where
    C: ValuePredictor<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.critic.predict_v(s)
    }
}

impl<S, C, P> ActionValuePredictor<S, P::Action> for TDAC<C, P>
where
    C: ActionValuePredictor<S, P::Action>,
    P: Policy<S>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.critic.predict_qs(s)
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.critic.predict_qsa(s, a)
    }
}

impl<S, C, P> Controller<S, P::Action> for TDAC<C, P>
where
    P: ParameterisedPolicy<S>,
{
    fn sample_target(&mut self, s: &S) -> P::Action {
        self.policy.sample(s)
    }

    fn sample_behaviour(&mut self, s: &S) -> P::Action {
        self.policy.sample(s)
    }
}
