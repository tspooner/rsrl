use Parameter;
use agents::{Agent, LinearAgent, Controller};
use domains::Transition;
use fa::{Function, VFunction, QFunction, Projector, Projection, Linear};
use geometry::{Space, ActionSpace};
use policies::{Policy, Greedy};
use ndarray::Array2;
use std::marker::PhantomData;


/// Greedy GQ control algorithm.
///
/// Maei, Hamid R., et al. "Toward off-policy learning control with function approximation."
/// Proceedings of the 27th International Conference on Machine Learning (ICML-10). 2010.
pub struct GreedyGQ<S: Space, M: Projector<S>, P: Policy> {
    pub fa_theta: Linear<S, M>,
    pub fa_w: Linear<S, M>,

    pub policy: P,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: Space, M: Projector<S>, P: Policy> GreedyGQ<S, M, P> {
    pub fn new<T1, T2, T3>(fa_theta: Linear<S, M>,
                           fa_w: Linear<S, M>,
                           policy: P,
                           alpha: T1,
                           beta: T2,
                           gamma: T3)
                           -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>,
              T3: Into<Parameter>
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

impl<S: Space, M: Projector<S>, P: Policy> Agent<S> for GreedyGQ<S, M, P> {
    type Sample = Transition<S, ActionSpace>;

    fn handle_sample(&mut self, t: &Transition<S, ActionSpace>) {
        let a = t.action;
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_w.projector.project(s);
        let phi_ns = self.fa_w.projector.project(ns);

        let nqs = QFunction::evaluate_phi(&self.fa_theta, &phi_ns);
        let na = Greedy.sample(&nqs);

        let td_estimate = VFunction::evaluate_phi(&mut self.fa_w, &phi_s);
        let td_error = t.reward + self.gamma.value()*self.fa_theta.evaluate_action_phi(&phi_ns, na) -
            self.fa_theta.evaluate_action_phi(&phi_s, a);

        let phi_s = self.fa_w.projector.expand_projection(phi_s);
        let phi_ns = self.fa_w.projector.expand_projection(phi_ns);

        let update_q = td_error*&phi_s - self.gamma*td_estimate*phi_ns;
        let update_v = (td_error - td_estimate)*phi_s;

        VFunction::update_phi(&mut self.fa_w, &Projection::Dense(update_v), self.alpha*self.beta);
        self.fa_theta.update_action_phi(&Projection::Dense(update_q), a, self.alpha.value());
    }

    fn handle_terminal(&mut self, _: &S::Repr) {
        self.alpha = self.alpha.step();
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal();
    }
}

impl<S: Space, M: Projector<S>, P: Policy> LinearAgent<S> for GreedyGQ<S, M, P> {
    fn weights(&self) -> Array2<f64> {
        self.fa_theta.weights.clone()
    }
}

impl<S: Space, M: Projector<S>, P: Policy> Controller<S, ActionSpace> for GreedyGQ<S, M, P> {
    fn pi(&mut self, s: &S::Repr) -> usize {
        self.evaluate_policy(&mut Greedy, s)
    }

    fn mu(&mut self, s: &S::Repr) -> usize {
        let qs: Vec<f64> = self.fa_theta.evaluate(s);

        self.policy.sample(qs.as_slice())
    }

    fn evaluate_policy<T: Policy>(&self, p: &mut T, s: &S::Repr) -> usize {
        let qs: Vec<f64> = self.fa_theta.evaluate(s);

        p.sample(qs.as_slice())
    }
}
