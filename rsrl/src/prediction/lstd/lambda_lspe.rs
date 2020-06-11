use crate::{
    domains::Batch,
    fa::linear::{basis::Basis, Features},
    utils::pinv,
    Handler,
    Parameterised,
};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Solve;
use std::ops::MulAssign;

#[derive(Debug, Parameterised)]
pub struct LambdaLSPE<B> {
    pub basis: B,
    #[weights]
    pub theta: Array1<f64>,

    pub alpha: f64,
    pub gamma: f64,
    pub lambda: f64,

    a: Array2<f64>,
    b: Array1<f64>,
    delta: f64,
}

impl<B: spaces::Space> LambdaLSPE<B> {
    pub fn new(basis: B, alpha: f64, gamma: f64, lambda: f64) -> Self {
        let n_features: usize = basis.dim().into();

        LambdaLSPE {
            basis,
            theta: Array1::zeros(n_features),

            alpha,
            gamma,
            lambda,

            a: Array2::eye(n_features) * 1e-6,
            b: Array1::zeros(n_features),
            delta: 0.0,
        }
    }
}

impl<B> LambdaLSPE<B> {
    fn solve(&mut self) {
        // First try the clean approach otherwise solve via SVD:
        let theta = self
            .a
            .solve(&self.b)
            .or_else(|_| pinv(&self.a).map(|ainv| ainv.dot(&self.b)));

        if let Ok(theta) = theta {
            self.theta.mul_assign(1.0 - self.alpha);
            self.theta.scaled_add(self.alpha, &theta);

            self.a.fill(0.0);
            self.b.fill(0.0);
            self.delta = 0.0;
        }
    }
}

impl<'m, S, A, B> Handler<&'m Batch<S, A>> for LambdaLSPE<B>
where B: Basis<&'m S, Value = Features>
{
    type Response = ();
    type Error = crate::fa::linear::Error;

    fn handle(&mut self, batch: &'m Batch<S, A>) -> Result<(), Self::Error> {
        for t in batch.iter().rev() {
            let (s, ns) = t.states();
            let phi_s = self.basis.project(s)?;

            self.delta *= self.gamma * self.lambda;

            if t.terminated() {
                let phi_s = phi_s.into_dense();

                self.b.scaled_add(self.delta + t.reward, &phi_s);
                self.a += &phi_s
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&(phi_s.view().insert_axis(Axis(0))));
                self.delta = 0.0;
            } else {
                let theta_s = phi_s.dot(&self.theta);
                let theta_ns = self.basis.project(ns)?.dot(&self.theta);

                let phi_s = phi_s.into_dense();
                let residual = t.reward + self.gamma * theta_ns - theta_s;

                self.delta += residual;
                self.b.scaled_add(theta_s + self.delta, &phi_s);
                self.a += &phi_s
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&(phi_s.view().insert_axis(Axis(0))));
            };
        }

        self.solve();

        Ok(())
    }
}
