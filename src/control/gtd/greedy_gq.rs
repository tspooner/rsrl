use crate::core::*;
use crate::domains::Transition;
use crate::fa::*;
use crate::geometry::{MatrixView, MatrixViewMut};
use crate::policies::{fixed::Greedy, Policy, FinitePolicy};

/// Greedy GQ control algorithm.
///
/// Maei, Hamid R., et al. "Toward off-policy learning control with function
/// approximation." Proceedings of the 27th International Conference on Machine
/// Learning (ICML-10). 2010.
pub struct GreedyGQ<Q, W, P> {
    pub fa_q: Shared<Q>,
    pub fa_w: Shared<W>,

    pub policy: Shared<P>,
    pub target: Greedy<Q>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<Q, W, P> GreedyGQ<Q, W, P> {
    pub fn new<P1, P2, P3>(
        fa_q: Shared<Q>,
        fa_w: Shared<W>,
        policy: Shared<P>,
        alpha: P1,
        beta: P2,
        gamma: P3,
    ) -> Self
    where
        P1: Into<Parameter>,
        P2: Into<Parameter>,
        P3: Into<Parameter>,
    {
        GreedyGQ {
            fa_q: fa_q.clone(),
            fa_w,

            policy,
            target: Greedy::new(fa_q),

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<Q, W, P> Algorithm for GreedyGQ<Q, W, P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.beta = self.beta.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, Q, W, P> OnlineLearner<S, P::Action> for GreedyGQ<Q, W, P>
where
    Q: QFunction<S>,
    W: VFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let phi_s = self.fa_w.to_features(s);
        let estimate = self.fa_w.evaluate(&phi_s).unwrap();

        if t.terminated() {
            let residual = t.reward - self.fa_q.evaluate_index(&phi_s, t.action).unwrap();

            self.fa_w.borrow_mut().update(
                &phi_s,
                self.alpha * self.beta * (residual - estimate)
            ).ok();
            self.fa_q.borrow_mut().update_index(
                &phi_s,
                t.action,
                self.alpha.value() * residual
            ).ok();
        } else {
            let ns = t.from.state();
            let na = self.sample_target(ns);
            let phi_ns = self.fa_w.to_features(ns);

            let residual =
                t.reward
                + self.gamma.value() * self.fa_q.evaluate_index(&phi_ns, na).unwrap()
                - self.fa_q.evaluate_index(&phi_s, t.action).unwrap();

            let n_features = self.fa_q.n_features();
            let update_q = residual * phi_s.clone().expanded(n_features)
                - estimate * self.gamma.value() * phi_ns.expanded(n_features);

            self.fa_w.borrow_mut().update(
                &phi_s,
                self.alpha * self.beta * (residual - estimate)
            ).ok();
            self.fa_q.borrow_mut().update_index(
                &Features::Dense(update_q),
                t.action,
                self.alpha.value()
            ).ok();
        }
    }
}

impl<S, Q, W, P> ValuePredictor<S> for GreedyGQ<Q, W, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.target.probabilities(s))
    }
}

impl<S, Q, W, P> ActionValuePredictor<S, P::Action> for GreedyGQ<Q, W, P>
where
    Q: QFunction<S>,
    P: FinitePolicy<S>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.fa_q.evaluate(&self.fa_q.to_features(s)).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: usize) -> f64 {
        self.fa_q.evaluate_index(&self.fa_q.to_features(s), a).unwrap()
    }
}

impl<S, Q, W, P> Controller<S, P::Action> for GreedyGQ<Q, W, P>
where
    Q: QFunction<S>,
    P: Policy<S, Action = <Greedy<Q> as Policy<S>>::Action>,
{
    fn sample_target(&mut self, s: &S) -> P::Action { self.target.sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<Q: Parameterised, W, P> Parameterised for GreedyGQ<Q, W, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_q.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.fa_q.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }
}
