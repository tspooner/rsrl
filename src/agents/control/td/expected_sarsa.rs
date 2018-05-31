use agents::{Controller, Predictor};
use domains::Transition;
use fa::QFunction;
use policies::Policy;
use std::marker::PhantomData;
use {Handler, Parameter, Vector};

/// Action probability-weighted variant of SARSA (aka "summation Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - van Seijen, H., van Hasselt, H., Whiteson, S., Wiering, M. (2009). A
/// theoretical and empirical analysis of Expected Sarsa. In Proceedings of the
/// IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning,
/// pp. 177–184.
pub struct ExpectedSARSA<S, Q: QFunction<S>, P: Policy> {
    pub q_func: Q,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, Q, P> ExpectedSARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    pub fn new<T1, T2>(q_func: Q, policy: P, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        ExpectedSARSA {
            q_func: q_func,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, Q, P> Handler<Transition<S, usize>> for ExpectedSARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let qs = self.q_func.evaluate(s).unwrap();
        let nqs = self.q_func.evaluate(ns).unwrap();

        let a = t.action;

        let pi: Vector<f64> = self.policy.probabilities(nqs.as_slice().unwrap()).into();
        let exp_nqs = pi.dot(&nqs);
        let td_error = t.reward + self.gamma * exp_nqs - qs[a];

        self.q_func.update_action(s, a, self.alpha * td_error);
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S, Q, P> Controller<S, usize> for ExpectedSARSA<S, Q, P>
where
    Q: QFunction<S>,
    P: Policy,
{
    fn pi(&mut self, s: &S) -> usize {
        self.policy
            .sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        p.sample(self.q_func.evaluate(s).unwrap().as_slice().unwrap())
    }
}

impl<S, Q: QFunction<S>, P: Policy> Predictor<S> for ExpectedSARSA<S, Q, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.q_func.evaluate(s).unwrap();
        let pi: Vector<f64> = self.policy.probabilities(nqs.as_slice().unwrap()).into();

        pi.dot(&nqs)
    }
}
