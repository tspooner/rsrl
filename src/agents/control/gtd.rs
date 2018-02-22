use {Parameter, Vector};
use agents::{Agent, Controller};
use domains::Transition;
use fa::{Approximator, SimpleLinear, MultiLinear, Projection, Projector, VFunction, QFunction};
use geometry::{ActionSpace, Space};
use policies::{Greedy, Policy};
use std::marker::PhantomData;

/// Greedy GQ control algorithm.
///
/// Maei, Hamid R., et al. "Toward off-policy learning control with function approximation."
/// Proceedings of the 27th International Conference on Machine Learning (ICML-10). 2010.
pub struct GreedyGQ<S: Space, M: Projector<S::Repr>, P: Policy> {
    pub fa_theta: MultiLinear<S::Repr, M>,
    pub fa_w: SimpleLinear<S::Repr, M>,

    pub policy: P,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, M: Projector<S::Repr>, P: Policy> GreedyGQ<S, M, P> {
    pub fn new<T1, T2, T3>(
        fa_theta: MultiLinear<S::Repr, M>,
        fa_w: SimpleLinear<S::Repr, M>,
        policy: P,
        alpha: T1,
        beta: T2,
        gamma: T3,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        GreedyGQ {
            fa_theta: fa_theta,
            fa_w: fa_w,

            policy: policy,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S: Space, M: Projector<S::Repr>, P: Policy> Agent for GreedyGQ<S, M, P> {
    type Sample = Transition<S, ActionSpace>;

    fn handle_sample(&mut self, t: &Transition<S, ActionSpace>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_w.projector.project(s);
        let phi_ns = self.fa_w.projector.project(ns);

        let nqs = self.fa_theta.evaluate_phi(&phi_ns);
        let na = Greedy.sample(nqs.as_slice().unwrap());

        let td_estimate = self.fa_w.evaluate_phi(&phi_s);
        let td_error = t.reward
            + self.gamma.value() * self.fa_theta.evaluate_action_phi(&phi_ns, na)
            - self.fa_theta.evaluate_action_phi(&phi_s, a);

        let phi_s = phi_s.expanded(self.fa_w.projector.span());
        let phi_ns = phi_ns.expanded(self.fa_w.projector.span());

        let update_q = td_error * phi_s.clone() - self.gamma * td_estimate * phi_ns;
        let update_v = (td_error - td_estimate) * phi_s;

        self.fa_w.update_phi(&Projection::Dense(update_v), self.alpha * self.beta);
        self.fa_theta.update_action_phi(&Projection::Dense(update_q), a, self.alpha.value());
    }

    fn handle_terminal(&mut self, _: &Self::Sample) {
        self.alpha = self.alpha.step();
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S: Space, M: Projector<S::Repr>, P: Policy> Controller<S, ActionSpace> for GreedyGQ<S, M, P> {
    fn pi(&mut self, s: &S::Repr) -> usize { self.evaluate_policy(&mut Greedy, s) }

    fn mu(&mut self, s: &S::Repr) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        self.policy.sample(qs.as_slice().unwrap())
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize {
        let qs: Vector<f64> = self.fa_theta.evaluate(s).unwrap();

        p.sample(qs.as_slice().unwrap())
    }
}
