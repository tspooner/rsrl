use agents::{Controller, Predictor, memory::Trace};
use domains::Transition;
use fa::{Approximator, MultiLFA, Projection, Projector, QFunction};
use policies::{Greedy, Policy};
use std::marker::PhantomData;
use {Handler, Parameter, Vector};

/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
pub struct QLambda<S, M: Projector<S>, P: Policy> {
    trace: Trace,

    pub fa_theta: MultiLFA<S, M>,
    pub policy: P,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, M: Projector<S>, P: Policy> QLambda<S, M, P> {
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
        QLambda {
            trace: trace,

            fa_theta: fa_theta,
            policy: policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: Policy> Handler<Transition<S, usize>> for QLambda<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.projector.project(s);
        let phi_ns = self.fa_theta.projector.project(ns);

        let qs = self.fa_theta.evaluate_phi(&phi_s);
        let nqs = self.fa_theta.evaluate_phi(&phi_ns);
        let na = Greedy.sample(nqs.as_slice().unwrap());

        let td_error = t.reward + self.gamma * nqs[na] - qs[a];

        if a == Greedy.sample(qs.as_slice().unwrap()) {
            let rate = self.trace.lambda.value() * self.gamma.value();
            self.trace.decay(rate);
        } else {
            self.trace.decay(0.0);
        }

        self.trace
            .update(&phi_s.expanded(self.fa_theta.projector.dim()));
        self.fa_theta.update_action_phi(
            &Projection::Dense(self.trace.get()),
            a,
            td_error * self.alpha,
        );
    }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.trace.decay(0.0);
        self.policy.handle_terminal();
    }
}

impl<S, M: Projector<S>, P: Policy> Controller<S, usize> for QLambda<S, M, P> {
    fn pi(&mut self, s: &S) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        Greedy.sample(qs.as_slice().unwrap())
    }

    fn mu(&mut self, s: &S) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        self.policy.sample(qs.as_slice().unwrap())
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        p.sample(qs.as_slice().unwrap())
    }
}

impl<S, M: Projector<S>, P: Policy> Predictor<S> for QLambda<S, M, P> {
    fn predict(&mut self, s: &S) -> f64 {
        let nqs = self.fa_theta.evaluate(s).unwrap();
        let na = Greedy.sample(nqs.as_slice().unwrap());

        nqs[na]
    }
}
