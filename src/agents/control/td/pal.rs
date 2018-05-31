use agents::{Controller, Predictor};
use domains::Transition;
use fa::QFunction;
use policies::{Greedy, Policy};
use std::marker::PhantomData;
use {Handler, Parameter};

/// Persistent Advantage Learning
///
/// # References
/// - Bellemare, Marc G., et al. "Increasing the Action Gap: New Operators for
/// Reinforcement Learning." AAAI. 2016.
pub struct PAL<S, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        PAL {
            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, Q, P> Handler<Transition<S, usize>> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s).unwrap();
        let nqs = self.q_func.evaluate(ns).unwrap();

        let a = t.action;
        let na = Greedy.sample(nqs.as_slice().unwrap());

        let vs = qs[Greedy.sample(qs.as_slice().unwrap())];

        let td_error = t.reward + self.gamma * nqs[na] - qs[a];
        let al_error = td_error - self.alpha * (vs - qs[a]);
        let pal_error = al_error.max(td_error - self.alpha * (vs - nqs[a]));

        self.q_func.update_action(s, a, self.alpha * pal_error);
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S, Q, P> Controller<S, usize> for PAL<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    fn pi(&mut self, s: &S) -> usize {
        Greedy.sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize {
        self.policy
            .sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        p.sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }
}

impl<S, Q: QFunction<S>, P: Policy> Predictor<S> for PAL<S, Q, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.q_func.evaluate(s).unwrap();
        let na = Greedy.sample(nqs.as_slice().unwrap());

        nqs[na]
    }
}
