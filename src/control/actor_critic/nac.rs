use crate::core::*;
use crate::domains::Transition;
use crate::fa::Parameterised;
use crate::geometry::norms::l1;
use crate::policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

/// Natural actor-critic.
pub struct NAC<C, P> {
    pub critic: Shared<C>,
    pub policy: Shared<P>,

    pub alpha: Parameter,
}

impl<C, P> NAC<C, P> {
    pub fn new<T: Into<Parameter>>(critic: Shared<C>, policy: Shared<P>, alpha: T) -> Self {
        NAC {
            critic,
            policy,

            alpha: alpha.into(),
        }
    }
}

impl<C, P> Algorithm for NAC<C, P>
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

impl<S, C, P> OnlineLearner<S, P::Action> for NAC<C, P>
where
    C: OnlineLearner<S, P::Action> + Parameterised,
    P: ParameterisedPolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        self.critic.borrow_mut().handle_transition(t);
        self.policy.borrow_mut().update_raw(
            self.alpha.value() * self.critic.borrow().weights()
        );
    }

    fn handle_sequence(&mut self, seq: &[Transition<S, P::Action>]) {
        self.critic.borrow_mut().handle_sequence(seq);
        self.policy.borrow_mut().update_raw(
            self.alpha.value() * self.critic.borrow().weights()
        );
    }
}

impl<S, C, P> ValuePredictor<S> for NAC<C, P>
where
    C: ValuePredictor<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.critic.borrow_mut().predict_v(s)
    }
}

impl<S, C, P> ActionValuePredictor<S, P::Action> for NAC<C, P>
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

impl<S, C, P> Controller<S, P::Action> for NAC<C, P>
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
