use core::{Algorithm, Controller, Predictor, Shared, Parameter, Vector, Matrix};
use domains::Transition;
use fa::{
    Approximator,
    Parameterised,
    VFunction,
    QFunction,

    Projection,
    Projector,
    SimpleLFA,
    MultiLFA
};
use policies::{fixed::Greedy, Policy, FinitePolicy};
use std::marker::PhantomData;

/// Greedy GQ control algorithm.
///
/// Maei, Hamid R., et al. "Toward off-policy learning control with function
/// approximation." Proceedings of the 27th International Conference on Machine
/// Learning (ICML-10). 2010.
pub struct GreedyGQ<S, M: Projector<S>, P: Policy<S>> {
    pub fa_theta: Shared<MultiLFA<S, M>>,
    pub fa_w: Shared<SimpleLFA<S, M>>,

    pub policy: Shared<P>,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: 'static, M: Projector<S> + 'static, P: Policy<S>> GreedyGQ<S, M, P> {
    pub fn new<T1, T2, T3>(
        fa_theta: Shared<MultiLFA<S, M>>,
        fa_w: Shared<SimpleLFA<S, M>>,
        policy: Shared<P>,
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
            fa_theta: fa_theta.clone(),
            fa_w: fa_w,

            policy: policy,
            target: Greedy::new(fa_theta),

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> Algorithm<S, P::Action> for GreedyGQ<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_w.borrow().projector.project(s);
        let phi_ns = self.fa_w.borrow().projector.project(ns);

        let na = self.sample_target(ns);

        let td_estimate = self.fa_w.borrow().evaluate_phi(&phi_s);
        let td_error = t.reward
            + self.gamma.value() * self.fa_theta.borrow().evaluate_action_phi(&phi_ns, na)
            - self.fa_theta.borrow().evaluate_action_phi(&phi_s, t.action);

        let phi_s = phi_s.expanded(self.fa_w.borrow().projector.dim());
        let phi_ns = phi_ns.expanded(self.fa_w.borrow().projector.dim());

        let update_q = td_error * phi_s.clone() - self.gamma * td_estimate * phi_ns;
        let update_v = (td_error - td_estimate) * phi_s;

        self.fa_w.borrow_mut()
            .update_phi(&Projection::Dense(update_v), self.alpha * self.beta);
        self.fa_theta.borrow_mut()
            .update_action_phi(&Projection::Dense(update_q), t.action, self.alpha.value());
    }

    fn handle_terminal(&mut self, t: &Transition<S, P::Action>) {
        self.alpha = self.alpha.step();
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();

        self.policy.borrow_mut().handle_terminal(t);
        self.target.handle_terminal(t);
    }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> Controller<S, P::Action> for GreedyGQ<S, M, P> {
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> Predictor<S, P::Action> for GreedyGQ<S, M, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.target.probabilities(s))
    }

    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.fa_theta.borrow().evaluate_action(&s, a)
    }
}

impl<S, M: Projector<S>, P: Policy<S, Action = usize>> Parameterised for GreedyGQ<S, M, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
