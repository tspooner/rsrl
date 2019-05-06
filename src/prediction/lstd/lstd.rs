use crate::{
    core::*,
    domains::Transition,
    fa::{Approximator, VFunction, Parameterised},
    geometry::{Space, Matrix, MatrixView, MatrixViewMut},
    utils::{argmaxima, pinv},
};
use ndarray::Axis;
use ndarray_linalg::solve::Solve;

#[derive(Parameterised)]
pub struct LSTD<F> {
    #[weights] pub fa_theta: F,

    pub gamma: Parameter,

    a: Matrix<f64>,
    b: Vector<f64>,
}

impl<F: Parameterised> LSTD<F> {
    pub fn new<T: Into<Parameter>>(fa_theta: F, gamma: T) -> Self {
        let dim = fa_theta.weights_dim();

        LSTD {
            fa_theta,

            gamma: gamma.into(),

            a: Matrix::zeros(dim),
            b: Vector::zeros(dim.0),
        }
    }
}

impl<F: Parameterised> LSTD<F> {
    pub fn solve(&mut self) {
        let mut w = self.fa_theta.weights_view_mut();

        if let Ok(theta) = self.a.solve(&self.b) {
            // First try the clean approach:
            w.assign(&theta);
        } else if let Ok(ainv) = pinv(&self.a) {
            // Otherwise solve via SVD:
            w.assign(&ainv.dot(&self.b));
        }
    }
}

impl<F> Algorithm for LSTD<F> {
    fn handle_terminal(&mut self) {
        self.gamma = self.gamma.step();
    }
}

impl<S, A, F: VFunction<S> + Parameterised> BatchLearner<S, A> for LSTD<F> {
    fn handle_batch(&mut self, ts: &[Transition<S, A>]) {
        ts.into_iter().for_each(|ref t| {
            let phi_s = self.fa_theta
                .embed(t.from.state())
                .expanded(self.a.rows());
            let pd = if t.terminated() {
                phi_s.clone()
            } else {
                let phi_ns = self.fa_theta
                    .embed(t.to.state())
                    .expanded(self.a.rows());

                phi_s.clone() - self.gamma.value() * phi_ns
            }.insert_axis(Axis(0));

            self.b.scaled_add(t.reward, &phi_s);
            self.a += &phi_s.insert_axis(Axis(1)).dot(&pd);
        });

        self.solve();
    }
}

impl<S, F: VFunction<S>> ValuePredictor<S> for LSTD<F> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(&self.fa_theta.embed(s)).unwrap()
    }
}

impl<S, A, F: VFunction<S>> ActionValuePredictor<S, A> for LSTD<F> {}
