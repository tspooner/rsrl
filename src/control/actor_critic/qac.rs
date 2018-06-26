use core::{Algorithm, Controller, Predictor, Parameter, Shared};
use domains::Transition;
use policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

/// Action-value actor-critic.
pub struct QAC<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: Policy<S>,
{
    pub critic: C,
    pub policy: Shared<P>,

    pub alpha: Parameter,

    phantom: PhantomData<S>,
}

impl<S, C, P> QAC<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: Policy<S>,
{
    pub fn new<T: Into<Parameter>>(critic: C, policy: Shared<P>, alpha: T) -> Self {
        QAC {
            critic: critic,
            policy: policy,

            alpha: alpha.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Clone, C, P> Algorithm<S, P::Action> for QAC<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: ParameterisedPolicy<S>,
    P::Action: Copy,
{
    fn handle_sample(&mut self, t: &Transition<S, P::Action>) {
        self.critic.handle_sample(t);

        let s = t.from.state();
        let qsa = self.critic.predict_qsa(s, t.action);

        self.policy.borrow_mut().update(s, t.action, self.alpha * qsa);
    }

    fn handle_terminal(&mut self, t: &Transition<S, P::Action>) {
        self.alpha = self.alpha.step();

        self.critic.handle_terminal(t);
        self.policy.borrow_mut().handle_terminal(t);
    }
}

impl<S: Clone, C, P> Controller<S, P::Action> for QAC<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: ParameterisedPolicy<S>,
    P::Action: Copy,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}
