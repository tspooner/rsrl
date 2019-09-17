use crate::{
    Algorithm, OnlineLearner, Parameter,
    control::Controller,
    domains::Transition,
    fa::Parameterised,
    policies::{Policy, DifferentiablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
};
use rand::Rng;

/// Natural actor-critic.
pub struct NAC<C, P> {
    pub critic: C,
    pub policy: P,

    pub alpha: Parameter,

    n_updates: usize,
}

impl<C, P> NAC<C, P> {
    pub fn new<T: Into<Parameter>>(critic: C, policy: P, alpha: T) -> Self {
        NAC {
            critic,
            policy,

            alpha: alpha.into(),

            n_updates: 0,
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

        self.critic.handle_terminal();
        self.policy.handle_terminal();
    }
}

impl<C, P> NAC<C, P> {
    fn update_policy<S>(&mut self)
    where
        C: Parameterised,
        P: DifferentiablePolicy<S>,
    {
        if self.n_updates % 100 == 0 {
            let pw_dim = self.policy.weights_dim();
            let n_features = pw_dim[0] * pw_dim[1];

            let cw = self.critic.weights_view();
            let grad = cw.slice(s![0..n_features, ..]).into_shape(pw_dim).unwrap();
            let norm = grad.fold(0.0, |acc, g| acc + g * g).sqrt().max(1e-3);

            self.policy.update_grad_scaled(&grad, self.alpha.value() / norm);
        }
    }
}

impl<S, C, P> OnlineLearner<S, P::Action> for NAC<C, P>
where
    C: OnlineLearner<S, P::Action> + Parameterised,
    P: DifferentiablePolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        self.critic.handle_transition(t);
        self.n_updates += 1;

        self.update_policy();
    }

    fn handle_sequence(&mut self, seq: &[Transition<S, P::Action>]) {
        self.critic.handle_sequence(seq);
        self.n_updates += seq.len();

        self.update_policy();
    }
}

impl<S, C, P> ValuePredictor<S> for NAC<C, P>
where
    C: ValuePredictor<S>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.critic.predict_v(s)
    }
}

impl<S, C, P> ActionValuePredictor<S, P::Action> for NAC<C, P>
where
    C: ActionValuePredictor<S, P::Action>,
    P: Policy<S>,
{
    fn predict_qsa(&self, s: &S, a: P::Action) -> f64 {
        self.critic.predict_qsa(s, a)
    }
}

impl<S, C, P> Controller<S, P::Action> for NAC<C, P>
where
    P: Policy<S>,
{
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}
