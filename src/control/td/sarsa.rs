use crate::{
    core::*,
    domains::Transition,
    fa::{
        Parameterised, Weights, WeightsView, WeightsViewMut,
        StateActionFunction, FiniteActionFunction
    },
    policies::{Policy, FinitePolicy},
};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use rand::{thread_rng, Rng};

/// On-policy variant of Watkins' Q-learning (aka "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
#[derive(Parameterised)]
pub struct SARSA<Q, P> {
    #[weights] pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<Q, P> SARSA<Q, P> {
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        SARSA {
            q_func,
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<Q, P: Algorithm> Algorithm for SARSA<Q, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for SARSA<Q, P>
where
    Q: StateActionFunction<S, P::Action, Output = f64>,
    P: Policy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let qsa = self.q_func.evaluate(s, &t.action);

        let residual = if t.terminated() {
            t.reward - qsa
        } else {
            let ns = t.to.state();
            let na = self.policy.sample(&mut thread_rng(), ns);
            let nqsna = self.q_func.evaluate(ns, &na);

            t.reward + self.gamma * nqsna - qsa
        };

        self.q_func.update(s, &t.action, self.alpha * residual);
    }
}

impl<S, Q, P: Policy<S>> Controller<S, P::Action> for SARSA<Q, P> {
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}

impl<S, Q, P> ValuePredictor<S> for SARSA<Q, P>
where
    Q: FiniteActionFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.q_func.evaluate_all(s).into_iter()
            .zip(self.policy.probabilities(s).into_iter())
            .fold(0.0, |acc, (q, p)| acc + q * p)
    }
}

impl<S, Q, P> ActionValuePredictor<S, P::Action> for SARSA<Q, P>
where
    Q: StateActionFunction<S, P::Action, Output = f64>,
    P: Policy<S>,
{
    fn predict_qsa(&self, s: &S, a: P::Action) -> f64 {
        self.q_func.evaluate(s, &a)
    }
}
