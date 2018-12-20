use crate::core::*;
use crate::domains::Transition;
use crate::policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

/// TD-error actor-critic.
pub struct TDAC<S, C, P> {
    pub critic: Shared<C>,
    pub policy: Shared<P>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, C, P> TDAC<S, C, P> {
    pub fn new<T1, T2>(critic: Shared<C>, policy: Shared<P>, alpha: T1, gamma: T2) -> Self
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

impl<S, C, P> Algorithm for TDAC<S, C, P>
where
    C: Algorithm,
    P: Algorithm,
{
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.critic.borrow_mut().handle_terminal();
        self.policy.borrow_mut().handle_terminal();
    }
}

impl<S, C, P> TDAC<S, C, P>
where
    C: ValuePredictor<S>,
    P: ParameterisedPolicy<S>,
    P::Action: Clone,
{
    fn update_policy(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let v = self.critic.borrow_mut().predict_v(s);
        let nv = self.critic.borrow_mut().predict_v(ns);

        let residual = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * nv - v
        };

        self.policy.borrow_mut().update(s, t.action.clone(), self.alpha * residual);
    }
}

impl<S, C, P> OnlineLearner<S, P::Action> for TDAC<S, C, P>
where
    C: OnlineLearner<S, P::Action> + ValuePredictor<S>,
    P: ParameterisedPolicy<S>,
    P::Action: Clone,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        self.critic.borrow_mut().handle_transition(t);
        self.update_policy(t);
    }

    fn handle_sequence(&mut self, sequence: &[Transition<S, P::Action>]) {
        self.critic.borrow_mut().handle_sequence(sequence);

        sequence.into_iter().for_each(|ref t| {
            self.update_policy(t);
        });
    }
}

impl<S, C, P> ValuePredictor<S> for TDAC<S, C, P>
where
    C: ValuePredictor<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.critic.borrow_mut().predict_v(s)
    }
}

impl<S, C, P> ActionValuePredictor<S, P::Action> for TDAC<S, C, P>
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

impl<S, C, P> Controller<S, P::Action> for TDAC<S, C, P>
where
    P: ParameterisedPolicy<S>,
{
    fn sample_target(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }

    fn sample_behaviour(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }
}
