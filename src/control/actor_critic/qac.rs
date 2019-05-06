use crate::core::*;
use crate::domains::Transition;
use crate::policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

/// Action-value actor-critic.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QAC<C, P> {
    pub critic: C,
    pub policy: P,

    pub alpha: Parameter,
}

impl<C, P> QAC<C, P> {
    pub fn new<T: Into<Parameter>>(critic: C, policy: P, alpha: T) -> Self {
        QAC {
            critic,
            policy,

            alpha: alpha.into(),
        }
    }
}

impl<C, P> Algorithm for QAC<C, P>
where
    C: Algorithm,
    P: Algorithm,
{
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();

        self.critic.handle_terminal();
        self.policy.handle_terminal();
    }
}

impl<C, P> QAC<C, P> {
    pub fn update_policy<S>(&mut self, t: &Transition<S, P::Action>)
    where
        C: OnlineLearner<S, P::Action> + ActionValuePredictor<S, P::Action>,
        P: ParameterisedPolicy<S>,
        P::Action: Clone,
    {
        let s = t.from.state();
        let qsa = self.critic.predict_qsa(s, t.action.clone());

        self.policy.update(s, t.action.clone(), self.alpha * qsa);
    }
}

impl<S, C, P> OnlineLearner<S, P::Action> for QAC<C, P>
where
    C: OnlineLearner<S, P::Action> + ActionValuePredictor<S, P::Action>,
    P: ParameterisedPolicy<S>,
    P::Action: Clone,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        self.critic.handle_transition(t);

        self.update_policy(t);
    }

    fn handle_sequence(&mut self, seq: &[Transition<S, P::Action>]) {
        self.critic.handle_sequence(seq);

        for t in seq {
            self.update_policy(t);
        }
    }
}

impl<S, C, P> ValuePredictor<S> for QAC<C, P>
where
    C: ValuePredictor<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.critic.predict_v(s)
    }
}

impl<S, C, P> ActionValuePredictor<S, P::Action> for QAC<C, P>
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

impl<S, C, P> Controller<S, P::Action> for QAC<C, P>
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
