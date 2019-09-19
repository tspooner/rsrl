use crate::{
    OnlineLearner, Shared, make_shared,
    control::Controller,
    domains::Transition,
    fa::{
        Parameterised, Weights, WeightsView, WeightsViewMut,
        StateActionFunction, EnumerableStateActionFunction,
    },
    policies::{Greedy, Policy, EnumerablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
};
use rand::{Rng, thread_rng};

/// Persistent Advantage Learning
///
/// # References
/// - Bellemare, Marc G., et al. "Increasing the Action Gap: New Operators for
/// Reinforcement Learning." AAAI. 2016.
#[derive(Parameterised)]
pub struct PAL<Q, P> {
    #[weights] pub q_func: Q,

    pub policy: P,
    pub target: Greedy<Q>,

    pub alpha: f64,
    pub gamma: f64,
}

impl<Q, P> PAL<Shared<Q>, P> {
    pub fn new(q_func: Q, policy: P, alpha: f64, gamma: f64) -> Self {
        let q_func = make_shared(q_func);

        PAL {
            q_func: q_func.clone(),

            policy,
            target: Greedy::new(q_func),

            alpha,
            gamma,
        }
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for PAL<Q, P>
where
    Q: EnumerableStateActionFunction<S>,
    P: EnumerablePolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let residual = if t.terminated() {
            t.reward - self.q_func.evaluate(s, &t.action)
        } else {
            let ns = t.to.state();
            let qs = self.q_func.evaluate_all(s);
            let nqs = self.q_func.evaluate_all(ns);

            let mut rng = thread_rng();
            let a_star = self.sample_target(&mut rng, s);
            let na_star = self.sample_target(&mut rng, ns);

            let td_error = t.reward + self.gamma * nqs[a_star] - qs[t.action];
            let al_error = td_error - self.alpha * (qs[a_star] - qs[t.action]);

            al_error.max(td_error - self.alpha * (nqs[na_star] - nqs[t.action]))
        };

        self.q_func.update(s, &t.action, self.alpha * residual);
    }
}

impl<S, Q, P> Controller<S, P::Action> for PAL<Q, P>
where
    Q: EnumerableStateActionFunction<S>,
    P: EnumerablePolicy<S>,
{
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.target.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}

impl<S, Q, P> ValuePredictor<S> for PAL<Q, P>
where
    Q: EnumerableStateActionFunction<S>,
    P: EnumerablePolicy<S>,
{
    fn predict_v(&self, s: &S) -> f64 { self.predict_qsa(s, self.target.mpa(s)) }
}

impl<S, Q, P> ActionValuePredictor<S, P::Action> for PAL<Q, P>
where
    Q: StateActionFunction<S, P::Action, Output = f64>,
    P: Policy<S>,
{
    fn predict_qsa(&self, s: &S, a: P::Action) -> f64 {
        self.q_func.evaluate(s, &a)
    }
}
