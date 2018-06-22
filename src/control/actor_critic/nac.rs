use core::{Algorithm, Controller, Predictor, Parameter, Shared};
use domains::Transition;
use fa::Parameterised;
use geometry::norms::l1;
use policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

/// Natural actor-critic.
pub struct NAC<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: Policy<S>,
{
    pub critic: C,
    pub policy: Shared<P>,

    pub alpha: Parameter,

    phantom: PhantomData<S>,
}

impl<S, C, P> NAC<S, C, P>
where
    C: Predictor<S, P::Action>,
    P: Policy<S>,
{
    pub fn new<T: Into<Parameter>>(critic: C, policy: Shared<P>, alpha: T) -> Self {
        NAC {
            critic: critic,
            policy: policy,

            alpha: alpha.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Clone, C, P> Algorithm<S, P::Action> for NAC<S, C, P>
where
    C: Predictor<S, P::Action> + Parameterised,
    P: ParameterisedPolicy<S>,
{
    fn handle_sample(&mut self, t: &Transition<S, P::Action>) {
        self.critic.handle_sample(t);

        let w = self.critic.weights();
        let z = l1(w.as_slice().unwrap());

        self.policy.borrow_mut().update_raw(self.alpha.value() * w / z);
    }

    fn handle_terminal(&mut self, t: &Transition<S, P::Action>) {
        self.alpha = self.alpha.step();

        self.critic.handle_terminal(t);
        self.policy.borrow_mut().handle_terminal(t);
    }
}

impl<S: Clone, C, P> Controller<S, P::Action> for NAC<S, C, P>
where
    C: Predictor<S, P::Action> + Parameterised,
    P: ParameterisedPolicy<S>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}
