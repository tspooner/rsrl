use crate::core::*;
use crate::domains::Transition;
use crate::policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

/// Continuous Actor-Critic Learning Automaton
pub struct CACLA<C, P> {
    pub critic: Shared<C>,
    pub policy: Shared<P>,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<C, P> CACLA<C, P> {
    pub fn new<T1, T2>(critic: Shared<C>, policy: Shared<P>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        CACLA {
            critic,
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<C, P> Algorithm for CACLA<C, P>
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

impl<S, C, P> OnlineLearner<S, P::Action> for CACLA<C, P>
where
    C: OnlineLearner<S, P::Action> + ValuePredictor<S>,
    P: ParameterisedPolicy<S, Action = f64>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let mut critic = self.critic.borrow_mut();
        let mut policy = self.policy.borrow_mut();

        let s = t.from.state();
        let v = critic.predict_v(s);
        let target = if t.terminated() {
            t.reward
        } else {
            t.reward + self.gamma * critic.predict_v(t.to.state())
        };

        critic.handle_transition(t);

        if target > v {
            let mpa = policy.mpa(s);

            policy.update(s, t.action, self.alpha * (t.action - mpa));
        }
    }
}

impl<S, C, P> ValuePredictor<S> for CACLA<C, P>
where
    C: ValuePredictor<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.critic.borrow_mut().predict_v(s)
    }
}

impl<S, C, P> ActionValuePredictor<S, P::Action> for CACLA<C, P>
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

use rand::{
    distributions::{Distribution, Normal as NormalDist},
    rngs::ThreadRng,
    thread_rng,
};

impl<S, C, P> Controller<S, P::Action> for CACLA<C, P>
where
    P: ParameterisedPolicy<S, Action = f64>,
{
    fn sample_target(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }

    fn sample_behaviour(&mut self, s: &S) -> P::Action {
        NormalDist::new(self.policy.borrow_mut().sample(s), 5.0).sample(&mut thread_rng())
    }
}
