use crate::{
    Handler, Function, Enumerable, Parameterised,
    domains::Transition,
    fa::StateActionUpdate,
    policies::{Policy, EnumerablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
};
use std::ops::Index;

/// Action probability-weighted variant of SARSA (aka "summation Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - van Seijen, H., van Hasselt, H., Whiteson, S., Wiering, M. (2009). A
/// theoretical and empirical analysis of Expected Sarsa. In Proceedings of the
/// IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning,
/// pp. 177–184.
#[derive(Parameterised)]
pub struct ExpectedSARSA<Q, P> {
    #[weights] pub q_func: Q,
    pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
}

impl<Q, P> ExpectedSARSA<Q, P> {
    pub fn new(q_func: Q, policy: P, alpha: f64, gamma: f64) -> Self {
        ExpectedSARSA {
            q_func,
            policy,

            alpha,
            gamma,
        }
    }
}

impl<'m, S, Q, P> Handler<&'m Transition<S, usize>> for ExpectedSARSA<Q, P>
where
    Q: Enumerable<(&'m S,)> + Handler<StateActionUpdate<&'m S, usize, f64>>,
    P: EnumerablePolicy<&'m S>,

    <Q as Function<(&'m S,)>>::Output: Index<usize, Output = f64> + IntoIterator<Item = f64>,
    <<Q as Function<(&'m S,)>>::Output as IntoIterator>::IntoIter: ExactSizeIterator,

    <P as Function<(&'m S,)>>::Output: Index<usize, Output = f64> + IntoIterator<Item = f64>,
    <<P as Function<(&'m S,)>>::Output as IntoIterator>::IntoIter: ExactSizeIterator,
{
    type Response = Q::Response;
    type Error = Q::Error;

    fn handle(&mut self, t: &'m Transition<S, usize>) -> Result<Self::Response, Self::Error> {
        let s = t.from.state();
        let qsa = self.q_func.evaluate_index((s,), t.action);
        let residual = if t.terminated() {
            t.reward - qsa
        } else {
            let ns = t.to.state();
            let exp_nv = self.predict_v(ns);

            t.reward + self.gamma * exp_nv - qsa
        };

        self.q_func.handle(StateActionUpdate {
            state: s,
            action: t.action,
            error: self.alpha * residual,
        })
    }
}

impl<S, Q, P> ValuePredictor<S> for ExpectedSARSA<Q, P>
where
    S: Clone,

    Q: Enumerable<(S,)>,
    P: EnumerablePolicy<S>,

    <Q as Function<(S,)>>::Output: Index<usize, Output = f64> + IntoIterator<Item = f64>,
    <<Q as Function<(S,)>>::Output as IntoIterator>::IntoIter: ExactSizeIterator,

    <P as Function<(S,)>>::Output: Index<usize, Output = f64> + IntoIterator<Item = f64>,
    <<P as Function<(S,)>>::Output as IntoIterator>::IntoIter: ExactSizeIterator,
{
    fn predict_v(&self, s: S) -> f64 {
        self.q_func.evaluate((s.clone(),)).into_iter()
                .zip(self.policy.evaluate((s,)).into_iter())
                .fold(0.0, |acc, (q, p)| acc + q * p)
    }
}

impl<S, Q, P> ActionValuePredictor<S, P::Action> for ExpectedSARSA<Q, P>
where
    Q: Function<(S, P::Action), Output = f64>,
    P: Policy<S>,
{
    fn predict_q(&self, s: S, a: P::Action) -> f64 {
        self.q_func.evaluate((s, a))
    }
}
