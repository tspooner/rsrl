use crate::{
    domains::Transition,
    fa::ScaledGradientUpdate,
    policies::Policy,
    traces,
    Differentiable,
    Function,
    Handler,
    Parameterised,
};
use rand::thread_rng;

/// On-policy variant of Watkins' Q-learning with eligibility traces (aka
/// "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
#[derive(Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct SARSALambda<Q, P, T> {
    #[weights]
    pub fa_theta: Q,
    pub policy: P,
    pub trace: T,

    pub alpha: f64,
    pub gamma: f64,
}

impl<Q, P, T> SARSALambda<Q, P, T> {
    pub fn new(fa_theta: Q, policy: P, trace: T, alpha: f64, gamma: f64) -> Self {
        SARSALambda {
            fa_theta,
            policy,
            trace,

            alpha,
            gamma,
        }
    }
}

type Tr<S, A, Q, R> = traces::Trace<<Q as Differentiable<(S, A)>>::Jacobian, R>;

impl<'m, S, Q, P, R> Handler<&'m Transition<S, P::Action>> for SARSALambda<
    Q, P,
    Tr<&'m S, &'m P::Action, Q, R>
>
where
    Q: Function<(&'m S, P::Action), Output = f64>
        + Differentiable<(&'m S, &'m P::Action), Output = f64>,
    Q: for<'j> Handler<
        ScaledGradientUpdate<&'j Tr<&'m S, &'m P::Action, Q, R>>,
    >,
    P: Policy<&'m S>,
    R: traces::UpdateRule<<Q as Differentiable<(&'m S, &'m P::Action)>>::Jacobian>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<Self::Response, Self::Error> {
        let s = t.from.state();
        let qsa = self.fa_theta.evaluate((s, &t.action));

        // Update trace with latest feature vector:
        self.trace.update(&self.fa_theta.grad((s, &t.action)));

        // Update weight vectors:
        if t.terminated() {
            self.fa_theta
                .handle(ScaledGradientUpdate {
                    alpha: self.alpha * (t.reward - qsa),
                    jacobian: &self.trace,
                })
                .ok();
            self.trace.reset();
        } else {
            let ns = t.to.state();
            let na = self.policy.sample(&mut thread_rng(), ns);
            let nqsna = self.fa_theta.evaluate((ns, na));

            self.fa_theta
                .handle(ScaledGradientUpdate {
                    alpha: self.alpha * (t.reward + self.gamma * nqsna - qsa),
                    jacobian: &self.trace,
                })
                .ok();
        };

        Ok(())
    }
}
