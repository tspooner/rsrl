use crate::{
    core::*,
    domains::Transition,
    fa::{
        Parameterised, Weights, WeightsView, WeightsViewMut,
        StateActionFunction, FiniteActionFunction,
    },
    policies::{Greedy, Policy, FinitePolicy},
};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use rand::{thread_rng, Rng};

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

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<Q, P> QLearning<Shared<Q>, P> {
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        let q_func = make_shared(q_func);

        QLearning {
            q_func: q_func.clone(),

            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<Q, P: Algorithm> Algorithm for QLearning<Q, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for QLearning<Q, P>
where
    Q: FiniteActionFunction<S>,
    P: FinitePolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let qsa = self.q_func.evaluate(s, &t.action);

        let residual = if t.terminated() {
            t.reward - qsa
        } else {
            let ns = t.to.state();
            let (_, nqsna) = self.q_func.evaluate_max(ns);

            t.reward + self.gamma * nqsna - qsa
        };

        self.q_func.update(s, &t.action, self.alpha * residual);
    }
}

impl<S, Q, P> Controller<S, P::Action> for QLearning<Q, P>
where
    Q: FiniteActionFunction<S>,
    P: FinitePolicy<S>,
{
    fn sample_target(&self, _: &mut impl Rng, s: &S) -> P::Action {
        self.q_func.evaluate_max(s).0
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}

impl<S, Q, P> ValuePredictor<S> for QLearning<Q, P>
where
    Q: FiniteActionFunction<S, Output = f64>,
    P: Policy<S>,
{
    fn predict_v(&self, s: &S) -> f64 { self.q_func.evaluate_max(s).1 }
}

impl<S, Q, P> ActionValuePredictor<S, <Greedy<Q> as Policy<S>>::Action> for QLearning<Q, P>
where
    Q: FiniteActionFunction<S, Output = f64>,
    P: Policy<S>,
{
    fn predict_qsa(&self, s: &S, a: <Greedy<Q> as Policy<S>>::Action) -> f64 {
        self.q_func.evaluate(s, &a)
    }
}
