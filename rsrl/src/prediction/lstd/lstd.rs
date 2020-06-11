use crate::{
    domains::Batch,
    fa::linear::{basis::Basis, Features},
    prediction::ValuePredictor,
    utils::pinv,
    Handler,
    Parameterised,
};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Solve;

#[derive(Debug, Parameterised)]
pub struct LSTD<B> {
    pub basis: B,
    #[weights]
    pub theta: Array1<f64>,

    pub gamma: f64,

    a: Array2<f64>,
    b: Array1<f64>,
}

impl<B: spaces::Space> LSTD<B> {
    pub fn new(basis: B, gamma: f64) -> Self {
        let n_features: usize = basis.dim().into();

        LSTD {
            basis,
            theta: Array1::zeros(n_features),

            gamma,

            a: Array2::eye(n_features) * 1e-6,
            b: Array1::zeros(n_features),
        }
    }
}

impl<B> LSTD<B> {
    pub fn solve(&mut self) {
        let theta = self
            .a
            .solve(&self.b)
            .or_else(|_| pinv(&self.a).map(|ainv| ainv.dot(&self.b)));

        if let Ok(theta) = theta {
            self.theta.assign(&theta);
        }
    }
}

impl<'m, S, A, B> Handler<&'m Batch<S, A>> for LSTD<B>
where B: Basis<&'m S, Value = Features>
{
    type Response = ();
    type Error = crate::fa::linear::Error;

    fn handle(&mut self, batch: &'m Batch<S, A>) -> Result<(), Self::Error> {
        for t in batch {
            let (s, ns) = t.states();

            let phi_s = self.basis.project(s)?.into_dense();

            self.b.scaled_add(t.reward, &phi_s);

            if t.terminated() {
                let phi_s = phi_s.insert_axis(Axis(1));

                self.a += &phi_s.view().dot(&phi_s.t());
            } else {
                let phi_ns = self.basis.project(ns)?.into_dense();
                let pd = (self.gamma * phi_ns - &phi_s).insert_axis(Axis(0));

                self.a -= &phi_s.insert_axis(Axis(1)).dot(&pd);
            }
        }

        self.solve();

        Ok(())
    }
}

impl<S, B: Basis<S, Value = Features>> ValuePredictor<S> for LSTD<B> {
    fn predict_v(&self, s: S) -> f64 { self.basis.project(s).unwrap().dot(&self.theta) }
}
