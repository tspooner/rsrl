use crate::{
    domains::Transition,
    fa::ScaledGradientUpdate,
    utils::argmax_first,
    traces,
    Differentiable,
    Enumerable,
    Function,
    Handler,
    Parameterised,
};
use std::ops::Index;

/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
#[derive(Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct QLambda<F, T> {
    #[weights]
    pub fa_theta: F,
    pub trace: T,

    pub alpha: f64,
    pub gamma: f64,
}

impl<Q, T> QLambda<Q, T> {
    pub fn new(fa_theta: Q, trace: T, alpha: f64, gamma: f64) -> Self {
        QLambda {
            fa_theta,
            trace,

            alpha,
            gamma,
        }
    }

    pub fn undiscounted(fa_theta: Q, trace: T, alpha: f64) -> Self {
        QLambda::new(fa_theta, trace, alpha, 1.0)
    }
}

type Tr<S, A, Q, R> = traces::Trace<<Q as Differentiable<(S, A)>>::Jacobian, R>;

impl<'m, S, Q, R> Handler<&'m Transition<S, usize>> for QLambda<Q, Tr<&'m S, usize, Q, R>>
where
    Q: Enumerable<(&'m S,)> + Differentiable<(&'m S, usize)>,
    Q: for<'j> Handler<ScaledGradientUpdate<&'j Tr<&'m S, usize, Q, R>>>,

    R: traces::UpdateRule<<Q as Differentiable<(&'m S, usize)>>::Jacobian>,

    <Q as Function<(&'m S,)>>::Output: Index<usize, Output = f64> + IntoIterator<Item = f64>,
    <<Q as Function<(&'m S,)>>::Output as IntoIterator>::IntoIter: ExactSizeIterator,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, t: &'m Transition<S, usize>) -> Result<(), ()> {
        let s = t.from.state();

        let qs = self.fa_theta.evaluate((s,));
        let qsa = qs[t.action];
        let grad_s = self.fa_theta.grad((s, t.action));

        // Update trace:
        if t.action != argmax_first(qs).0 { self.trace.reset(); }

        self.trace.update(&grad_s);

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
            let (_, nqs_max) = self.fa_theta.find_max((ns,));

            self.fa_theta
                .handle(ScaledGradientUpdate {
                    alpha: self.alpha * (t.reward + self.gamma * nqs_max - qsa),
                    jacobian: &self.trace,
                })
                .ok();
        };

        Ok(())
    }
}
