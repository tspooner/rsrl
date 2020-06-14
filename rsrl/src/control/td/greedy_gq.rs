use crate::{
    domains::Transition,
    fa::StateActionUpdate,
    policies::Policy,
    Enumerable,
    Function,
    Handler,
};
use std::f64;

// TODO: Extract prediction component GQ / GQ(lambda) into seperate
// implementations.

#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Response<RQ, RT> {
    pub td_error: f64,

    pub q_response: RQ,
    pub td_response: RT,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub enum Error<EQ, ET> {
    QFuncError(EQ),
    TDEstError(ET),
}

/// Greedy GQ control algorithm.
///
/// Maei, Hamid R., et al. "Toward off-policy learning control with function
/// approximation." Proceedings of the 27th International Conference on Machine
/// Learning (ICML-10). 2010.
#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct GreedyGQ<Q, T, P> {
    #[weights]
    pub fa_q: Q,
    pub fa_td: T,

    pub behaviour_policy: P,

    pub gamma: f64,
}

impl<'m, S, Q, T, P> Handler<&'m Transition<S, P::Action>> for GreedyGQ<Q, T, P>
where
    Q: Handler<StateActionUpdate<&'m S, usize>> + Function<(&'m S,)> + Enumerable<(&'m S,)>,

    Q::Output: IntoIterator<Item = f64> + std::ops::Index<usize, Output = f64>,
    <Q::Output as IntoIterator>::IntoIter: ExactSizeIterator,

    T: Handler<StateActionUpdate<&'m S, usize>> + Function<(&'m S, usize), Output = f64>,

    P: Policy<&'m S, Action = usize>,
{
    type Response = Response<Q::Response, T::Response>;
    type Error = Error<Q::Error, T::Error>;

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<Self::Response, Self::Error> {
        let s = t.from.state();

        let qsa = self.fa_q.evaluate_index((s,), t.action);
        let td_est = self.fa_td.evaluate((s, t.action));

        if t.terminated() {
            let td_error = t.reward - qsa;

            let q_response = self.fa_q
                .handle(StateActionUpdate {
                    state: s,
                    action: t.action,
                    error: td_error,
                })
                .map_err(|e| Error::QFuncError(e))?;

            let td_response = self.fa_td
                .handle(StateActionUpdate {
                    state: s,
                    action: t.action,
                    error: td_error - td_est,
                })
                .map_err(|e| Error::TDEstError(e))?;

            Ok(Response {
                td_error,

                q_response,
                td_response,
            })
        } else {
            let ns = t.to.state();
            let (na, qnsna) = self.fa_q.find_max((ns,));

            let td_error = t.reward + self.gamma * qnsna - qsa;

            let q_response = self.fa_q
                .handle(StateActionUpdate {
                    state: s,
                    action: t.action,
                    error: td_error,
                })
                .and_then(|_| {
                    self.fa_q
                        .handle(StateActionUpdate {
                            state: ns,
                            action: na,
                            error: -self.gamma * td_est,
                        })
                })
                .map_err(|e| Error::QFuncError(e))?;

            let td_response = self.fa_td
                .handle(StateActionUpdate {
                    state: s,
                    action: t.action,
                    error: td_error - td_est,
                })
                .map_err(|e| Error::TDEstError(e))?;

            Ok(Response {
                td_error,

                q_response,
                td_response,
            })
        }
    }
}
