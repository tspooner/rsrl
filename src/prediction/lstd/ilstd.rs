use crate::{
    core::*,
    domains::Transition,
    fa::{Approximator, VFunction, Parameterised, Projector},
    geometry::{Space, Matrix, MatrixView, MatrixViewMut},
    utils::argmaxima,
};
use ndarray::Axis;

#[allow(non_camel_case_types)]
pub struct iLSTD<F> {
    pub fa_theta: Shared<F>,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub n_updates: usize,

    a: Matrix<f64>,
    mu: Vector<f64>,
}

impl<F: Parameterised> iLSTD<F> {
    pub fn new<T1, T2>(fa_theta: Shared<F>, n_updates: usize, alpha: T1, gamma: T2) -> Self
    where T1: Into<Parameter>,
            T2: Into<Parameter>
    {
        let dim = fa_theta.weights_dim();

        iLSTD {
            fa_theta,

            alpha: alpha.into(),
            gamma: gamma.into(),
            n_updates,

            a: Matrix::zeros(dim),
            mu: Vector::zeros(dim.0),
        }
    }
}

impl<F: Parameterised> iLSTD<F> {
    fn solve(&mut self) {
        let mut fa = self.fa_theta.borrow_mut();
        let alpha = self.alpha.value();

        for _ in 0..self.n_updates {
            let (_, idx) = argmaxima(self.mu.mapv(|v| v.abs()).as_slice().unwrap());

            for j in idx {
                unsafe {
                    let update = alpha * self.mu.uget(j);

                    *fa.weights_view_mut().uget_mut((j, 0)) += update;
                    self.mu.scaled_add(-update, &self.a.column(j));
                }
            }
        }
    }
}

impl<F> Algorithm for iLSTD<F> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, F: VFunction<S> + Parameterised> OnlineLearner<S, A> for iLSTD<F> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        // (D x 1)
        let phi_s = self.fa_theta
            .to_features(t.from.state())
            .expanded(self.a.rows());

        // (1 x D)
        let pd = if t.terminated() {
            phi_s.clone().insert_axis(Axis(0))
        } else {
            let phi_ns = self.fa_theta
                .to_features(t.to.state())
                .expanded(self.a.rows());

            (phi_s.clone() - self.gamma.value() * phi_ns).insert_axis(Axis(0))
        };

        // (D x D)
        let delta_a = phi_s.clone().insert_axis(Axis(1)).dot(&pd);

        self.a += &delta_a;

        self.mu.scaled_add(t.reward, &phi_s);
        self.mu -= &delta_a.dot(&self.fa_theta.weights_view());

        self.solve();
    }
}

impl<S, F: VFunction<S>> ValuePredictor<S> for iLSTD<F> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(&self.fa_theta.to_features(s)).unwrap()
    }
}

impl<S, A, F: VFunction<S>> ActionValuePredictor<S, A> for iLSTD<F> {}

impl<F: Parameterised> Parameterised for iLSTD<F> {
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
