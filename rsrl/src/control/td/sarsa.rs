use crate::{
    Handler, Function, Parameterised,
    domains::Transition,
    fa::StateActionUpdate,
    policies::Policy,
    prediction::{ValuePredictor, ActionValuePredictor},
};
use rand::thread_rng;

/// On-policy variant of Watkins' Q-learning (aka "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct SARSA<Q, P> {
    #[weights] pub q_func: Q,
    pub policy: P,

    pub gamma: f64,
}

impl<Q, P> SARSA<Q, P> {
    pub fn new(q_func: Q, policy: P, gamma: f64) -> Self {
        SARSA {
            q_func,
            policy,

            gamma,
        }
    }
}

impl<'m, S, Q, P> Handler<&'m Transition<S, P::Action>> for SARSA<Q, P>
where
    Q: Function<(&'m S, P::Action), Output = f64> +
        for<'a> Function<(&'m S, &'a P::Action), Output = f64> +
        Handler<StateActionUpdate<&'m S, &'m P::Action>>,
    P: Policy<&'m S>,
{
    type Response = Q::Response;
    type Error = Q::Error;

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<Self::Response, Self::Error> {
        let s = t.from.state();
        let qsa = self.q_func.evaluate((s, &t.action));

        let residual = if t.terminated() {
            t.reward - qsa
        } else {
            let ns = t.to.state();
            let na = self.policy.sample(&mut thread_rng(), ns);
            let nqsna = self.q_func.evaluate((ns, na));

            t.reward + self.gamma * nqsna - qsa
        };

        self.q_func.handle(StateActionUpdate {
            state: s,
            action: &t.action,
            error: residual,
        })
    }
}

impl<S, Q, P> ValuePredictor<S> for SARSA<Q, P>
where
    S: Clone,
    Q: Function<(S, P::Action), Output = f64>,
    P: Policy<S>,
{
    fn predict_v(&self, s: S) -> f64 {
        let a = self.policy.sample(&mut thread_rng(), s.clone());

        self.q_func.evaluate((s, a))
    }
}

impl<S, A, Q, P> ActionValuePredictor<S, A> for SARSA<Q, P>
where
    A: std::borrow::Borrow<P::Action>,
    Q: Function<(S, A), Output = f64>,
    P: Policy<S>,
{
    fn predict_q(&self, s: S, a: A) -> f64 {
        self.q_func.evaluate((s, a))
    }
}
