use crate::{
    OnlineLearner,
    control::Controller,
    domains::Transition,
    fa::{Weights, WeightsView, WeightsViewMut, Parameterised},
    policies::{Policy, DifferentiablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
};
use rand::Rng;

/// Off-policy TD-based actor-critic.
#[derive(Parameterised)]
pub struct OffPAC<C, T, B> {
    #[weights] pub critic: C,

    pub target: T,
    pub behaviour: B,

    pub alpha: f64,
    pub gamma: f64,
}

impl<C, T: Parameterised, B> OffPAC<C, T, B> {
    pub fn new(
        critic: C,
        target: T,
        behaviour: B,
        alpha: f64,
        gamma: f64,
    ) -> Self {
        OffPAC {
            critic,

            target,
            behaviour,

            alpha,
            gamma,
        }
    }
}

impl<C, T, B> OffPAC<C, T, B> {
    fn update_policy<S>(&mut self, t: &Transition<S, T::Action>)
    where
        C: ValuePredictor<S>,
        T: DifferentiablePolicy<S>,
        B: Policy<S, Action = T::Action>,
    {
        let (s, ns) = (t.from.state(), t.to.state());

        let v = self.critic.predict_v(s);

        let residual = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * self.critic.predict_v(ns) - v
        };
        let is_ratio = {
            let pi = self.target.probability(s, &t.action);
            let b = self.behaviour.probability(s, &t.action);

            pi / b
        };

        self.target.update(s, &t.action, self.alpha * residual * is_ratio);
    }
}

impl<S, C, T, B> OnlineLearner<S, T::Action> for OffPAC<C, T, B>
where
    C: OnlineLearner<S, T::Action> + ValuePredictor<S>,
    T: DifferentiablePolicy<S>,
    B: Policy<S, Action = T::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, T::Action>) {
        self.critic.handle_transition(t);

        self.update_policy(t);
    }

    fn handle_terminal(&mut self) {
        self.critic.handle_terminal();
    }
}

impl<S, C, T, B> ValuePredictor<S> for OffPAC<C, T, B>
where
    C: ValuePredictor<S>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.critic.predict_v(s)
    }
}

impl<S, C, T, B> ActionValuePredictor<S, T::Action> for OffPAC<C, T, B>
where
    C: ActionValuePredictor<S, T::Action>,
    T: Policy<S>,
{
    fn predict_qsa(&self, s: &S, a: T::Action) -> f64 {
        self.critic.predict_qsa(s, a)
    }
}

impl<S, C, T, B> Controller<S, T::Action> for OffPAC<C, T, B>
where
    T: DifferentiablePolicy<S>,
    B: Policy<S, Action = T::Action>,
{
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> T::Action {
        self.target.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> B::Action {
        self.behaviour.sample(rng, s)
    }
}
