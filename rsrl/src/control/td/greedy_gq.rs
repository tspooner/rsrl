use crate::{
    domains::Transition,
    fa::StateActionUpdate,
    policies::{EnumerablePolicy, Policy},
    Enumerable,
    Function,
    Handler,
};
use std::f64;

// TODO: Extract prediction component GQ / GQ(lambda) into seperate
// implementations.

/// Greedy GQ control algorithm.
///
/// Maei, Hamid R., et al. "Toward off-policy learning control with function
/// approximation." Proceedings of the 27th International Conference on Machine
/// Learning (ICML-10). 2010.
#[derive(Parameterised)]
pub struct GreedyGQ<Q, T, P> {
    #[weights]
    pub fa_q: Q,
    pub fa_td: T,

    pub behaviour_policy: P,

    pub gamma: f64,
}

impl<Q, T, P> GreedyGQ<Q, T, P> {
    pub fn new(fa_q: Q, fa_td: T, behaviour_policy: P, gamma: f64) -> Self {
        GreedyGQ {
            fa_q,
            fa_td,

            behaviour_policy,

            gamma,
        }
    }
}

impl<'m, S, Q, T, P> Handler<&'m Transition<S, P::Action>> for GreedyGQ<Q, T, P>
where
    Q: Handler<StateActionUpdate<&'m S, usize>> + Function<(&'m S,)> + Enumerable<(&'m S,)>,

    Q::Output: IntoIterator<Item = f64> + std::ops::Index<usize, Output = f64>,
    <Q::Output as IntoIterator>::IntoIter: ExactSizeIterator,

    T: Handler<StateActionUpdate<&'m S, usize>> + Function<(&'m S, usize), Output = f64>,

    P: Policy<&'m S, Action = usize>,
{
    type Response = ();
    type Error = ();

    fn handle(&mut self, t: &'m Transition<S, P::Action>) -> Result<(), ()> {
        let s = t.from.state();

        let qsa = self.fa_q.evaluate_index((s,), t.action);
        let td_est = self.fa_td.evaluate((s, t.action));

        if t.terminated() {
            let residual = t.reward - qsa;

            self.fa_q
                .handle(StateActionUpdate {
                    state: s,
                    action: t.action,
                    error: residual,
                })
                .ok();

            self.fa_td
                .handle(StateActionUpdate {
                    state: s,
                    action: t.action,
                    error: residual - td_est,
                })
                .ok();
        } else {
            let ns = t.to.state();
            let (na, qnsna) = self.fa_q.find_max((ns,));

            let residual = t.reward + self.gamma * qnsna - qsa;

            self.fa_q
                .handle(StateActionUpdate {
                    state: s,
                    action: t.action,
                    error: residual,
                })
                .ok();

            self.fa_q
                .handle(StateActionUpdate {
                    state: ns,
                    action: na,
                    error: -self.gamma * td_est,
                })
                .ok();

            self.fa_td
                .handle(StateActionUpdate {
                    state: s,
                    action: t.action,
                    error: residual - td_est,
                })
                .ok();
        }

        Ok(())
    }
}
