use crate::{
    core::*,
    domains::Transition,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction,
        linear::{Features, LinearStateFunction},
    },
    utils::{argmaxima, pinv},
};
use ndarray::Axis;
use ndarray_linalg::solve::Solve;
use std::ops::MulAssign;

#[derive(Parameterised)]
pub struct LambdaLSPE<F> {
    #[weights] pub fa_theta: F,

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

            a: Matrix::eye(dim[0]) * 1e-6,
            b: Vector::zeros(dim[0]),
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

impl<S, A, F> BatchLearner<S, A> for LambdaLSPE<F>
where
    F: LinearStateFunction<S, Output = f64>,
{
    fn handle_batch(&mut self, batch: &[Transition<S, A>]) {
        batch.into_iter().rev().for_each(|ref t| {
            let (s, ns) = t.states();
            let phi_s = self.fa_theta.features(s);

            self.delta *= self.gamma * self.lambda;

            if t.terminated() {
                let phi_s = phi_s.expanded();

                self.b.scaled_add(self.delta + t.reward, &phi_s);
                self.a +=
                    &phi_s.view().insert_axis(Axis(1))
                    .dot(&(phi_s.view().insert_axis(Axis(0))));
                self.delta = 0.0;
            } else {
                let theta_s = self.fa_theta.evaluate_features(&phi_s);
                let phi_s = phi_s.expanded();

                let theta_ns = self.fa_theta.evaluate(ns);
                let residual = t.reward + self.gamma * theta_ns - theta_s;

                self.delta += residual;
                self.b.scaled_add(theta_s + self.delta, &phi_s);
                self.a +=
                    &phi_s.view().insert_axis(Axis(1))
                    .dot(&(phi_s.view().insert_axis(Axis(0))));
            };
        });

        self.solve();
    }
}

impl<S, F> ValuePredictor<S> for LambdaLSPE<F>
where
    F: StateFunction<S, Output = f64>
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s)
    }
}
