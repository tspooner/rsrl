use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, VFunction, Parameterised, Projector, Projection, ScalarLFA};
use crate::geometry::Space;
use ndarray::Axis;
use crate::utils::argmaxima;

pub struct RecursiveLSTD<M> {
    pub fa_theta: Shared<ScalarLFA<M>>,
    pub gamma: Parameter,

    c_mat: Matrix<f64>,
}

impl<M: Space> RecursiveLSTD<M> {
    pub fn new<T: Into<Parameter>>(fa_theta: Shared<ScalarLFA<M>>, gamma: T) -> Self {
        let n_features = fa_theta.projector.dim();

        RecursiveLSTD {
            fa_theta,
            gamma: gamma.into(),

            c_mat: Matrix::eye(n_features)*1e-6,
        }
    }
}

impl<M> RecursiveLSTD<M> {
    #[inline(always)]
    fn expand_phi(&self, phi: Projection) -> /* (D x 1) */ Matrix<f64> {
        phi.expanded(self.c_mat.rows()).insert_axis(Axis(1))
    }
}

impl<M> Algorithm for RecursiveLSTD<M> {
    fn handle_terminal(&mut self) {
        self.gamma = self.gamma.step();
    }
}

impl<S, A, M: Projector<S>> OnlineLearner<S, A> for RecursiveLSTD<M> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let phi_s = self.fa_theta.projector.project(t.from.state());
        let v = self.fa_theta.evaluate_phi(&phi_s);
        let phi_s = self.expand_phi(phi_s);

        let (pd, residual) = if t.terminated() {
            (phi_s.clone(), t.reward - v)
        } else {
            let phi_ns = self.fa_theta.projector.project(t.to.state());
            let nv = self.fa_theta.evaluate_phi(&phi_ns);
            let phi_ns = self.expand_phi(phi_ns);

            (
                phi_s.clone() - self.gamma.value() * phi_ns,
                t.reward + self.gamma * nv - v
            )
        };

        // (1 x D) <- ((D x D) . (D x 1))T
        //  Note: self.permuted_axes([1, 0]) is equivalent to taking the transpose.
        let g = self.c_mat.dot(&pd).permuted_axes([1, 0]);

        // (1 x 1) <- (1 x D) . (D x 1)
        let a = 1.0 + unsafe { g.dot(&phi_s).uget((0, 0)) };

        // (D x 1) <- (D x D) . (D x 1)
        let v = self.c_mat.dot(&phi_s);

        // (D x D) <- (D x 1) . (1 x D)
        let vg = v.dot(&g);

        self.c_mat.scaled_add(-1.0 / a, &vg);
        self.fa_theta.borrow_mut().update_phi(&Projection::Dense(
            v.index_axis_move(Axis(1), 0)
        ), residual / a);
    }
}

impl<S, M> ValuePredictor<S> for RecursiveLSTD<M>
where
    ScalarLFA<M>: VFunction<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(s).unwrap()
    }
}

impl<S, A, M> ActionValuePredictor<S, A> for RecursiveLSTD<M>
where
    ScalarLFA<M>: VFunction<S>,
{}

impl<M> Parameterised for RecursiveLSTD<M>
where
    ScalarLFA<M>: Parameterised,
{
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }
}
