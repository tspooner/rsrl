use crate::{
    core::*,
    domains::Transition,
    fa::{Approximator, VFunction, Parameterised, Projector},
    geometry::{Space, Matrix, MatrixView, MatrixViewMut},
    utils::{argmaxima, pinv},
};
use ndarray::Axis;
use ndarray_linalg::solve::Solve;
use std::ops::MulAssign;

pub struct LambdaLSPE<F> {
    pub fa_theta: F,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub lambda: Parameter,

    a: Matrix<f64>,
    b: Vector<f64>,
    delta: f64,
}

impl<F: Parameterised> LambdaLSPE<F> {
    pub fn new<T1, T2, T3>(fa_theta: F, alpha: T1, gamma: T2, lambda: T3) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        let dim = fa_theta.weights_dim();

        LambdaLSPE {
            fa_theta,

            alpha: alpha.into(),
            gamma: gamma.into(),
            lambda: lambda.into(),

            a: Matrix::zeros(dim),
            b: Vector::zeros(dim.0),
            delta: 0.0,
        }
    }
}

impl<F: Parameterised> LambdaLSPE<F> {
    fn solve(&mut self) {
        // First try the clean approach otherwise solve via SVD:
        if let Ok(theta) = self.a.solve(&self.b).or_else(|_| {
            pinv(&self.a).map(|ainv| ainv.dot(&self.b))
        }) {
            let mut w = self.fa_theta.weights_view_mut();

            w.mul_assign(1.0 - self.alpha);
            w.scaled_add(self.alpha.value(), &theta);

            self.a.fill(0.0);
            self.b.fill(0.0);
            self.delta = 0.0;
        }
    }
}

impl<M> Algorithm for LambdaLSPE<M> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
        self.lambda = self.lambda.step();
    }
}

impl<S, A, F: VFunction<S> + Parameterised> BatchLearner<S, A> for LambdaLSPE<F> {
    fn handle_batch(&mut self, batch: &[Transition<S, A>]) {
        batch.into_iter().rev().for_each(|ref t| {
            let phi_s = self.fa_theta.to_features(t.from.state());
            let v = self.fa_theta.evaluate(&phi_s).unwrap();
            let phi_s = phi_s.expanded(self.a.rows());

            if t.terminated() {
                let residual = t.reward - v;

                self.b.scaled_add(v + self.gamma * self.lambda * self.delta + residual, &phi_s);
                self.a += &phi_s.clone().insert_axis(Axis(1)).dot(&(phi_s.insert_axis(Axis(0))));

            } else {
                let phi_ns = self.fa_theta.to_features(t.to.state());
                let residual =
                    t.reward + self.gamma * self.fa_theta.evaluate(&phi_ns).unwrap() - v;

                self.delta = self.gamma * self.lambda * self.delta + residual;

                self.b.scaled_add(v + self.delta, &phi_s);
                self.a += &phi_s.clone().insert_axis(Axis(1)).dot(&(phi_s.insert_axis(Axis(0))));
            };
        });

        self.solve();
    }
}

impl<S, F: VFunction<S>> ValuePredictor<S> for LambdaLSPE<F> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(&self.fa_theta.to_features(s)).unwrap()
    }
}

impl<S, A, F: VFunction<S>> ActionValuePredictor<S, A> for LambdaLSPE<F> {}

impl<F: Parameterised> Parameterised for LambdaLSPE<F> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.fa_theta.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.fa_theta.weights_view_mut()
    }
}
