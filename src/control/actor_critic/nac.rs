use core::*;
use domains::Transition;
use fa::Parameterised;
use geometry::norms::l1;
use policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

/// Natural actor-critic.
pub struct NAC<S, C, P> {
    pub critic: Shared<C>,
    pub policy: Shared<P>,

    pub alpha: Parameter,

    phantom: PhantomData<S>,
}

impl<S, C, P> NAC<S, C, P> {
    pub fn new<T: Into<Parameter>>(critic: Shared<C>, policy: Shared<P>, alpha: T) -> Self {
        NAC {
            critic,
            policy,

            alpha: alpha.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, C, P> Algorithm for NAC<S, C, P>
where
    C: Algorithm,
    P: Algorithm,
{
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();

        self.critic.borrow_mut().handle_terminal();
        self.policy.borrow_mut().handle_terminal();
    }
}

impl<S, C, P> OnlineLearner<S, P::Action> for NAC<S, C, P>
where
    C: OnlineLearner<S, P::Action> + Parameterised,
    P: ParameterisedPolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        self.critic.borrow_mut().handle_transition(t);

        let w = self.critic.borrow().weights();
        let z = l1(w.as_slice().unwrap());

        self.policy.borrow_mut().update_raw(self.alpha.value() * w / z);
    }
}

impl<S, C, P> ValuePredictor<S> for NAC<S, C, P>
where
    C: ValuePredictor<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.critic.borrow_mut().predict_v(s)
    }
}

impl<S, C, P> ActionValuePredictor<S, P::Action> for NAC<S, C, P>
where
    C: ActionValuePredictor<S, P::Action>,
    P: Policy<S>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.critic.borrow_mut().predict_qs(s)
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.critic.borrow_mut().predict_qsa(s, a)
    }
}

impl<S, C, P> Controller<S, P::Action> for NAC<S, C, P>
where
    P: Policy<S>,
{
    fn sample_target(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }

    fn sample_behaviour(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }
}
