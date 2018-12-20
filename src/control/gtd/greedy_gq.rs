use core::*;
use domains::Transition;
use fa::*;
use policies::{fixed::Greedy, Policy, FinitePolicy};

/// Greedy GQ control algorithm.
///
/// Maei, Hamid R., et al. "Toward off-policy learning control with function
/// approximation." Proceedings of the 27th International Conference on Machine
/// Learning (ICML-10). 2010.
pub struct GreedyGQ<S, M: Projector<S>, P> {
    pub fa_theta: Shared<MultiLFA<S, M>>,
    pub fa_w: Shared<SimpleLFA<S, M>>,

    pub policy: Shared<P>,
    pub target: Greedy<S>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<S, M, P> GreedyGQ<S, M, P>
where
    S: 'static,
    M: Projector<S> + 'static,
{
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
            fa_w,

            policy,
            target: Greedy::new(fa_theta),

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<S, M: Projector<S>, P> Algorithm for GreedyGQ<S, M, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, M, P> OnlineLearner<S, P::Action> for GreedyGQ<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = <Greedy<S> as Policy<S>>::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let dim = self.fa_theta.borrow().projector.dim();
        let phi_s = self.fa_w.borrow().projector.project(s);

        let estimate = self.fa_w.borrow().evaluate_phi(&phi_s);

        if t.terminated() {
            let residual = t.reward - self.fa_theta.borrow().evaluate_action_phi(&phi_s, t.action);

            self.fa_w.borrow_mut().update_phi(
                &phi_s,
                self.alpha * self.beta * (residual - estimate)
            );
            self.fa_theta.borrow_mut().update_action_phi(
                &phi_s,
                t.action,
                self.alpha.value() * residual
            );
        } else {
            let na = self.sample_target(ns);
            let phi_ns = self.fa_w.borrow().projector.project(ns);

            let residual =
                t.reward
                + self.gamma.value() * self.fa_theta.borrow().evaluate_action_phi(&phi_ns, na)
                - self.fa_theta.borrow().evaluate_action_phi(&phi_s, t.action);

            let update_q = residual * phi_s.clone().expanded(dim)
                - estimate * self.gamma.value() * phi_ns.expanded(dim);

            self.fa_w.borrow_mut().update_phi(
                &phi_s,
                self.alpha * self.beta * (residual - estimate)
            );
            self.fa_theta.borrow_mut().update_action_phi(
                &Projection::Dense(update_q),
                t.action,
                self.alpha.value()
            );
        }
    }
}

impl<S, M, P> ValuePredictor<S> for GreedyGQ<S, M, P>
where
    M: Projector<S>,
    P: FinitePolicy<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.target.probabilities(s))
    }
}

impl<S, M, P> ActionValuePredictor<S, P::Action> for GreedyGQ<S, M, P>
where
    M: Projector<S>,
    P: FinitePolicy<S>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.fa_theta.borrow().evaluate_action(&s, a)
    }
}

impl<S, M, P> Controller<S, P::Action> for GreedyGQ<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = <Greedy<S> as Policy<S>>::Action>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<S, M: Projector<S>, P> Parameterised for GreedyGQ<S, M, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
