use core::{Algorithm, Predictor, Parameter, Trace, Shared};
use domains::Transition;
use fa::{Approximator, VFunction, Parameterised, Projector, Projection, SimpleLFA};
use geometry::{Space, Vector, Matrix};
use ndarray::Axis;
use utils::argmaxima;


pub struct RecursiveLSTD<S, P: Projector<S>> {
    pub fa_theta: Shared<SimpleLFA<S, P>>,
    pub gamma: Parameter,

    c_mat: Matrix<f64>,
}

impl<S, P: Projector<S>> RecursiveLSTD<S, P> {
    pub fn new<T: Into<Parameter>>(fa_theta: Shared<SimpleLFA<S, P>>, gamma: T) -> Self {
        let n_features = fa_theta.borrow().projector.dim();

        RecursiveLSTD {
            fa_theta,
            gamma: gamma.into(),

            c_mat: Matrix::eye(n_features)*1e-6,
        }
    }
}

impl<S, P: Projector<S>> RecursiveLSTD<S, P> {
    #[inline(always)]
    fn expand_phi(&self, phi: Projection) -> /* (D x 1) */ Matrix<f64> {
        phi.expanded(self.c_mat.rows()).insert_axis(Axis(1))
    }

    #[inline(always)]
    fn update_matrices(&mut self, g: Matrix<f64>, a: f64, v: Matrix<f64>, residual: f64) {
        // (D x D) <- (D x 1) . (1 x D)
        let vg = v.dot(&g);

        self.c_mat.scaled_add(-1.0 / a, &vg);
        self.fa_theta.borrow_mut().update_phi(&Projection::Dense(
            v.index_axis_move(Axis(1), 0)
        ), residual / a);
    }

    #[inline]
    fn do_update(&mut self, phi_s: &Matrix<f64>, pd: &Matrix<f64>, residual: f64) {
        // (1 x D) <- ((D x D) . (D x 1))T
        //  Note: self.permuted_axes([1, 0]) is equivalent to taking the transpose.
        let g = self.c_mat.dot(pd).permuted_axes([1, 0]);

        // (1 x 1) <- (1 x D) . (D x 1)
        let a = 1.0 + unsafe { g.dot(phi_s).uget((0, 0)) };

        // (D x 1) <- (D x D) . (D x 1)
        let v = self.c_mat.dot(phi_s);

        self.update_matrices(g, a, v, residual);
    }
}

impl<S, A, P: Projector<S>> Algorithm<S, A> for RecursiveLSTD<S, P> {
    fn handle_sample(&mut self, t: &Transition<S, A>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.fa_theta.borrow().projector.project(s);
        let phi_ns = self.fa_theta.borrow().projector.project(ns);

        let residual =
            t.reward + self.gamma * self.fa_theta.borrow().evaluate_phi(&phi_ns) -
            self.fa_theta.borrow().evaluate_phi(&phi_s);

        let phi_s = self.expand_phi(phi_s);
        let pd = phi_s.clone() - self.gamma.value() * self.expand_phi(phi_ns);

        self.do_update(&phi_s, &pd, residual);
    }

    fn handle_terminal(&mut self, t: &Transition<S, A>) {
        {
            let phi_s = self.fa_theta.borrow().projector.project(t.from.state());
            let residual = t.reward - self.fa_theta.borrow().evaluate_phi(&phi_s);

            let phi_s = self.expand_phi(phi_s);

            self.do_update(&phi_s, &phi_s, residual);
        }

        self.gamma = self.gamma.step();
    }
}

impl<S, A, P: Projector<S>> Predictor<S, A> for RecursiveLSTD<S, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }
}

impl<S, P: Projector<S>> Parameterised for RecursiveLSTD<S, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
