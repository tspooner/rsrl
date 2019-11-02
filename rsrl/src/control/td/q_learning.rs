use crate::{
    OnlineLearner, Shared, make_shared,
    control::Controller,
    domains::Transition,
    fa::{
        EnumerableStateActionFunction,
        Parameterised, Weights, WeightsView, WeightsViewMut,
    },
    policies::{Greedy, Policy, EnumerablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
};
use rand::Rng;

/// Watkins' Q-learning.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
#[derive(Parameterised)]
pub struct QLearning<Q, P> {
    #[weights] pub q_func: Q,

    pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
}

impl<Q, P> QLearning<Shared<Q>, P> {
    pub fn new(q_func: Q, policy: P, alpha: f64, gamma: f64) -> Self {
        let q_func = make_shared(q_func);

        QLearning {
            q_func: q_func.clone(),

            policy,

            alpha,
            gamma,
        }
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for QLearning<Q, P>
where
    Q: EnumerableStateActionFunction<S>,
    P: EnumerablePolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let qsa = self.q_func.evaluate(s, &t.action);

        let residual = if t.terminated() {
            t.reward - qsa
        } else {
            let ns = t.to.state();
            let (_, nqsna) = self.q_func.find_max(ns);

            t.reward + self.gamma * nqsna - qsa
        };

        self.q_func.update(s, &t.action, self.alpha * residual);
    }
}

impl<S, Q, P> Controller<S, P::Action> for QLearning<Q, P>
where
    Q: EnumerableStateActionFunction<S>,
    P: EnumerablePolicy<S>,
{
    fn sample_target(&self, _: &mut impl Rng, s: &S) -> P::Action {
        self.q_func.find_max(s).0
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}

impl<S, Q, P> ValuePredictor<S> for QLearning<Q, P>
where
    Q: EnumerableStateActionFunction<S, Output = f64>,
    P: Policy<S>,
{
    fn predict_v(&self, s: &S) -> f64 { self.q_func.find_max(s).1 }
}

impl<S, Q, P> ActionValuePredictor<S, <Greedy<Q> as Policy<S>>::Action> for QLearning<Q, P>
where
    Q: EnumerableStateActionFunction<S, Output = f64>,
    P: Policy<S>,
{
    fn predict_qsa(&self, s: &S, a: <Greedy<Q> as Policy<S>>::Action) -> f64 {
        self.q_func.evaluate(s, &a)
    }
}
