use agents::Controller;
use domains::Transition;
use fa::{MultiLFA, Projection, Projector, QFunction, SimpleLFA, VFunction};
use policies::{fixed::Greedy, Policy, FinitePolicy};
use std::marker::PhantomData;
use {Shared, Handler, Parameter};

/// Greedy GQ control algorithm.
///
/// Maei, Hamid R., et al. "Toward off-policy learning control with function
/// approximation." Proceedings of the 27th International Conference on Machine
/// Learning (ICML-10). 2010.
pub struct GreedyGQ<S, M: Projector<S>, P: FinitePolicy<S>> {
    pub fa_theta: Shared<MultiLFA<S, M>>,
    pub fa_w: Shared<SimpleLFA<S, M>>,

    pub policy: P,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: 'static, M: Projector<S> + 'static, P: FinitePolicy<S>> GreedyGQ<S, M, P> {
    pub fn new<T1, T2, T3>(
        fa_theta: Shared<MultiLFA<S, M>>,
        fa_w: Shared<SimpleLFA<S, M>>,
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

impl<S, M: Projector<S>, P: FinitePolicy<S>> Handler<Transition<S, usize>> for GreedyGQ<S, M, P> {
    fn handle_sample(&mut self, t: &Transition<S, usize>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_w.borrow().projector.project(s);
        let phi_ns = self.fa_w.borrow().projector.project(ns);

        let na = self.mu(ns);

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

    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.alpha = self.alpha.step();
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();

        self.policy.handle_terminal(t);
        self.target.handle_terminal(t);
    }
}

impl<S, M: Projector<S>, P: FinitePolicy<S>> Controller<S, usize> for GreedyGQ<S, M, P> {
    fn pi(&mut self, s: &S) -> usize {
        self.target.sample(&s)
    }

    fn mu(&mut self, s: &S) -> usize {
        self.policy.sample(&s)
    }
}
