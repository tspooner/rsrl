use agents::{Controller, Predictor, memory::Trace};
use domains::Transition;
use fa::{Approximator, MultiLFA, Projection, Projector, QFunction};
use policies::Policy;
use std::marker::PhantomData;
use {Handler, Parameter, Vector};

/// On-policy variant of Watkins' Q-learning with eligibility traces (aka
/// "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
pub struct SARSALambda<S, M: Projector<S>, P: Policy> {
    trace: Trace,

    pub fa_theta: MultiLFA<S, M>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M: Projector<S>, P: Policy> SARSALambda<S, M, P> {
    pub fn new<T1, T2>(
        trace: Trace,
        fa_theta: MultiLFA<S, M>,
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

impl<S, M: Projector<S>, P: Policy> Handler<Transition<S, usize>> for SARSALambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.projector.project(s);
        let phi_ns = self.fa_theta.projector.project(ns);

        let qsa = self.fa_theta.evaluate_action_phi(&phi_s, a);
        let nqs = self.fa_theta.evaluate_phi(&phi_ns);
        let na = self.policy.sample(nqs.as_slice().unwrap());

        let rate = self.trace.lambda.value() * self.gamma.value();
        let td_error = t.reward + self.gamma * nqs[na] - qsa;

        self.trace.decay(rate);
        self.trace
            .update(&phi_s.expanded(self.fa_theta.projector.dim()));

        self.fa_theta.update_action_phi(
            &Projection::Dense(self.trace.get()),
            a,
            self.alpha * td_error,
        );
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
        self.policy.handle_terminal();
    }
}

impl<S, M: Projector<S>, P: Policy> Controller<S, usize> for SARSALambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        self.policy.sample(qs.as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize { self.pi(s) }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        p.sample(qs.as_slice().unwrap())
    }
}

impl<S, M: Projector<S>, P: Policy> Predictor<S> for SARSALambda<S, M, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.fa_theta.evaluate(s).unwrap();
        let pi: Vector<f64> = self.policy.probabilities(nqs.as_slice().unwrap()).into();

        pi.dot(&nqs)
    }
}
