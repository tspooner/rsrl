use core::{Algorithm, Controller, Predictor, Parameter, Shared};
use domains::Transition;
use policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

/// Advantage actor-critic.
pub struct A2C<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: Policy<S>,
{
    pub critic: C,
    pub policy: Shared<P>,

    pub alpha: Parameter,

    phantom: PhantomData<S>,
}

impl<S, C, P> A2C<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: Policy<S>,
{
    pub fn new<T: Into<Parameter>>(critic: C, policy: Shared<P>, alpha: T) -> Self {
        A2C {
            critic,
            policy,

            alpha: alpha.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, C, P> A2C<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: ParameterisedPolicy<S>,
    P::Action: Copy,
{
    #[inline(always)]
    fn update_policy(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let v = self.critic.predict_v(s);
        let qsa = self.critic.predict_qsa(s, t.action);

        self.policy.borrow_mut().update(s, t.action, self.alpha * (qsa - v));
    }
}

impl<S: Clone, C, P> Algorithm<S, P::Action> for A2C<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: ParameterisedPolicy<S>,
    P::Action: Copy,
{
    fn handle_sample(&mut self, t: &Transition<S, P::Action>) {
        self.critic.handle_sample(t);
        self.update_policy(t);
    }

    fn handle_terminal(&mut self, t: &Transition<S, P::Action>) {
        self.critic.handle_terminal(t);

        self.update_policy(t);
        self.policy.borrow_mut().handle_terminal(t);

        self.alpha = self.alpha.step();
    }
}

impl<S: Clone, C, P> Controller<S, P::Action> for A2C<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: ParameterisedPolicy<S>,
    P::Action: Copy,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}
