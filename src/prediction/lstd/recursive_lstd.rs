use crate::{
    core::*,
    domains::Transition,
    fa::{Approximator, VFunction, Parameterised, Projector, Features},
    geometry::{Space, Matrix, MatrixView, MatrixViewMut},
    utils::argmaxima,
};
use ndarray::Axis;

pub struct RecursiveLSTD<F> {
    pub fa_theta: Shared<F>,
    pub gamma: Parameter,

    c_mat: Matrix<f64>,
}

impl<F: Parameterised> RecursiveLSTD<F> {
    pub fn new<T: Into<Parameter>>(fa_theta: Shared<F>, gamma: T) -> Self {
        let n_features = fa_theta.weights_dim().0;

        RecursiveLSTD {
            fa_theta,
            gamma: gamma.into(),

            c_mat: Matrix::eye(n_features)*1e-6,
        }
    }
}

impl<F> RecursiveLSTD<F> {
    #[inline(always)]
    fn expand_phi(&self, phi: Features) -> /* (D x 1) */ Matrix<f64> {
        phi.expanded(self.c_mat.rows()).insert_axis(Axis(1))
    }
}

impl<F> Algorithm for RecursiveLSTD<F> {
    fn handle_terminal(&mut self) {
        self.gamma = self.gamma.step();
    }
}

impl<S, A, F: VFunction<S>> OnlineLearner<S, A> for RecursiveLSTD<F> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let phi_s = self.fa_theta.to_features(t.from.state());
        let v = self.fa_theta.evaluate(&phi_s).unwrap();
        let phi_s = self.expand_phi(phi_s);

        let (pd, residual) = if t.terminated() {
            (phi_s.clone(), t.reward - v)
        } else {
            let phi_ns = self.fa_theta.to_features(t.to.state());
            let nv = self.fa_theta.evaluate(&phi_ns).unwrap();
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
        self.fa_theta.borrow_mut().update(&Features::Dense(
            v.index_axis_move(Axis(1), 0)
        ), residual / a).ok();
    }
}

impl<S, F: VFunction<S>> ValuePredictor<S> for RecursiveLSTD<F> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(&self.fa_theta.to_features(s)).unwrap()
    }
}

impl<S, A, F: VFunction<S>> ActionValuePredictor<S, A> for RecursiveLSTD<F> {}

impl<F: Parameterised> Parameterised for RecursiveLSTD<F> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.fa_theta.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unimplemented!()
    }
}
