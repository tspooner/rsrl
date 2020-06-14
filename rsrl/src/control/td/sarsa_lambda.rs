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

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Response {
    td_error: f64,
}

/// On-policy variant of Watkins' Q-learning with eligibility traces (aka
/// "modified Q-learning").
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
pub struct SARSALambda<Q, P, T> {
    #[weights]
    pub fa_theta: Q,
    pub policy: P,
    pub trace: T,

    pub alpha: f64,
    pub gamma: f64,
}

type Tr<S, A, Q, R> = traces::Trace<<Q as Differentiable<(S, A)>>::Jacobian, R>;


impl<'m, S, Q, P, R> Handler<&'m Transition<S, P::Action>> for SARSALambda<
    Q, P, Tr<&'m S, &'m P::Action, Q, R>
>
where
    Q: Function<(&'m S, P::Action), Output = f64> +
        Differentiable<(&'m S, &'m P::Action), Output = f64> +
        for<'j> Handler<ScaledGradientUpdate<&'j Tr<&'m S, &'m P::Action, Q, R>>>,
    P: Policy<&'m S>,
    R: traces::UpdateRule<<Q as Differentiable<(&'m S, &'m P::Action)>>::Jacobian>,
{
    type Response = Response;
    type Error = ();

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<Self::Response, Self::Error> {
        let s = t.from.state();
        let qsa = self.fa_theta.evaluate((s, &t.action));

        // Update trace with latest feature vector:
        self.trace.update(&self.fa_theta.grad((s, &t.action)));

        // Update weight vectors:
        let td_error = if t.terminated() {
            let residual = t.reward - qsa;

            self.fa_theta.handle(ScaledGradientUpdate {
                alpha: self.alpha * residual,
                jacobian: &self.trace,
            }).map_err(|_| ())?;
            self.trace.reset();

            residual
        } else {
            let ns = t.to.state();
            let na = self.policy.sample(&mut thread_rng(), ns);
            let nqsna = self.fa_theta.evaluate((ns, na));

            let residual = t.reward + self.gamma * nqsna - qsa;

            self.fa_theta.handle(ScaledGradientUpdate {
                alpha: self.alpha * residual,
                jacobian: &self.trace,
            }).map_err(|_| ())?;

            residual
        };

        Ok(Response { td_error, })
    }
}
