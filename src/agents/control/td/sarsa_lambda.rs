use agents::{Controller, Predictor, memory::Trace};
use domains::Transition;
use fa::{Approximator, MultiLFA, Projection, Projector, QFunction};
use policies::FinitePolicy;
use std::marker::PhantomData;
use {Handler, Shared, Parameter};

/// On-policy variant of Watkins' Q-learning with eligibility traces (aka
/// "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
pub struct SARSALambda<S, M: Projector<S>, P: FinitePolicy<S>> {
    trace: Trace,

    pub fa_theta: Shared<MultiLFA<S, M>>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> SARSALambda<S, M, P> {
    pub fn new<T1, T2>(
        trace: Trace,
        fa_theta: Shared<MultiLFA<S, M>>,
        policy: P,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        SARSALambda {
            trace: trace,

            fa_theta: fa_theta,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Handler<Transition<S, usize>> for SARSALambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.borrow().projector.project(s);

        let qa = self.fa_theta.borrow().evaluate_action_phi(&phi_s, t.action);
        let nqa = self.fa_theta.borrow().evaluate_action(ns, self.policy.sample(ns));

        let rate = self.trace.lambda.value() * self.gamma.value();
        let td_error = t.reward + self.gamma * nqa - qa;

        self.trace.decay(rate);
        self.trace
            .update(&phi_s.expanded(self.fa_theta.borrow().projector.dim()));

        self.fa_theta.borrow_mut().update_action_phi(
            &Projection::Dense(self.trace.get()),
            t.action,
            self.alpha * td_error,
        );
    }

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);

        self.policy.handle_terminal(t);
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Controller<S, usize> for SARSALambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize { self.policy.sample(s) }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Predictor<S> for SARSALambda<S, M, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.fa_theta.borrow().evaluate(s).unwrap();
        let pi = self.policy.probabilities(s);

        pi.dot(&nqs)
    }
}
