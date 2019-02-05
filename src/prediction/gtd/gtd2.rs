use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, Parameterised, Projection, Projector, ScalarLFA, VFunction};
use crate::geometry::Space;

pub struct GTD2<M> {
    pub fa_theta: Shared<ScalarLFA<M>>,
    pub fa_w: Shared<ScalarLFA<M>>,

    pub alpha: Parameter,
    pub beta: Parameter,
    pub gamma: Parameter,
}

impl<M: Space> GTD2<M> {
    pub fn new<T1, T2, T3>(
        fa_theta: Shared<ScalarLFA<M>>,
        fa_w: Shared<ScalarLFA<M>>,
        alpha: T1,
        beta: T2,
        gamma: T3,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        if fa_theta.projector.dim() != fa_w.projector.dim() {
            panic!("fa_theta and fa_w must be equivalent function approximators.")
        }

        GTD2 {
            fa_theta,
            fa_w,

            alpha: alpha.into(),
            beta: beta.into(),
            gamma: gamma.into(),
        }
    }
}

impl<M> Algorithm for GTD2<M> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.beta = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, M: Projector<S>> OnlineLearner<S, A> for GTD2<M> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let (phi_s, phi_ns) = t.map_states(|s| self.fa_theta.projector.project(s));

        let v = self.fa_theta.evaluate_phi(&phi_s);

        let td_estimate = self.fa_w.evaluate_phi(&phi_s);
        let td_error = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * self.fa_theta.evaluate_phi(&phi_ns) - v
        };

        self.fa_w.borrow_mut().update_phi(&phi_s, self.beta * (td_error - td_estimate));

        let dim = self.fa_theta.projector.dim();
        let pd = phi_s.expanded(dim) - self.gamma.value() * phi_ns.expanded(dim);

        self.fa_theta.borrow_mut().update_phi(&Projection::Dense(pd), self.alpha * td_estimate);
    }
}

impl<S, M> ValuePredictor<S> for GTD2<M>
where
    ScalarLFA<M>: VFunction<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(s).unwrap()
    }
}

impl<S, A, M> ActionValuePredictor<S, A> for GTD2<M>
where
    ScalarLFA<M>: VFunction<S>,
{}

impl<M> Parameterised for GTD2<M>
where
    ScalarLFA<M>: Parameterised,
{
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }
}
