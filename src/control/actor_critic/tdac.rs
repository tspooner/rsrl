use core::{Algorithm, Controller, Predictor, Parameter, Shared};
use domains::Transition;
use policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

/// Action-value actor-critic.
pub struct TDAC<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: Policy<S>,
{
    pub critic: C,
    pub policy: Shared<P>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, C, P> TDAC<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: Policy<S>,
{
    pub fn new<T1, T2>(critic: C, policy: Shared<P>, alpha: T1, gamma: T2) -> Self
        where
            T1: Into<Parameter>,
            T2: Into<Parameter>,
    {
        TDAC {
            critic,
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, C, P> TDAC<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: ParameterisedPolicy<S>,
    P::Action: Copy,
{
    #[inline(always)]
    fn update_policy(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let v = self.critic.predict_v(s);
        let nv = self.critic.predict_v(ns);
        let td_error = t.reward + self.gamma*nv - v;

        self.policy.borrow_mut().update(s, t.action, self.alpha * td_error);
    }
}

impl<S: Clone, C, P> Algorithm<S, P::Action> for TDAC<S, C, P>
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
        {
            self.critic.handle_terminal(t);

            self.update_policy(t);
            self.policy.borrow_mut().handle_terminal(t);
        }

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S: Clone, C, P> Controller<S, P::Action> for TDAC<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: ParameterisedPolicy<S>,
    P::Action: Copy,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}
