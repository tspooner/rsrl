use crate::{
    domains::Transition,
    fa::StateActionUpdate,
    utils::argmax_first,
    Enumerable,
    Function,
    Handler,
    Parameterised,
};
use std::ops::Index;

/// Persistent Advantage Learning
///
/// # References
/// - Bellemare, Marc G., et al. "Increasing the Action Gap: New Operators for
/// Reinforcement Learning." AAAI. 2016.
#[derive(Parameterised)]
pub struct PAL<Q> {
    #[weights]
    pub q_func: Q,

    pub alpha: f64,
    pub gamma: f64,
}

impl<Q> PAL<Q> {
    pub fn new(q_func: Q, alpha: f64, gamma: f64) -> Self {
        PAL {
            q_func,

            alpha,
            gamma,
        }
    }
}

impl<'m, S, Q> Handler<&'m Transition<S, usize>> for PAL<Q>
where
    Q: Enumerable<(&'m S,), Output = Vec<f64>> + Handler<StateActionUpdate<&'m S, usize, f64>>,
    <Q as Function<(&'m S,)>>::Output: Index<usize, Output = f64> + IntoIterator,
    <<Q as Function<(&'m S,)>>::Output as IntoIterator>::IntoIter: ExactSizeIterator,
{
    type Response = Q::Response;
    type Error = Q::Error;

    fn handle(&mut self, t: &'m Transition<S, usize>) -> Result<Self::Response, Self::Error> {
        let s = t.from.state();

        let residual = if t.terminated() {
            t.reward - self.q_func.evaluate_index((s,), t.action)
        } else {
            let ns = t.to.state();
            let qs = self.q_func.evaluate((s,));
            let nqs = self.q_func.evaluate((ns,));

            let a_star = argmax_first(qs.iter().copied()).0;
            let na_star = argmax_first(nqs.iter().copied()).0;

            let td_error = t.reward + self.gamma * nqs[a_star] - qs[t.action];
            let al_error = td_error - self.alpha * (qs[a_star] - qs[t.action]);

            al_error.max(td_error - self.alpha * (nqs[na_star] - nqs[t.action]))
        };

        self.q_func.handle(StateActionUpdate {
            state: s,
            action: t.action,
            error: self.alpha * residual,
        })
    }
}
