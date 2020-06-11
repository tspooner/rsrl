use crate::{
    domains::Batch,
    fa::linear::{basis::Basis, Features},
    utils::pinv,
    Handler,
    Parameterised,
};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Solve;

#[derive(Debug, Parameterised)]
pub struct LSTDLambda<B> {
    pub basis: B,
    #[weights]
    pub theta: Array1<f64>,

    pub gamma: f64,
    pub lambda: f64,

    a: Array2<f64>,
    b: Array1<f64>,
    z: Array1<f64>,
}

impl<B: spaces::Space> LSTDLambda<B> {
    pub fn new(basis: B, gamma: f64, lambda: f64) -> Self {
        let n_features: usize = basis.dim().into();

        LSTDLambda {
            basis,
            theta: Array1::zeros(n_features),

            gamma,
            lambda,

            a: Array2::eye(n_features) * 1e-6,
            b: Array1::zeros(n_features),
            z: Array1::zeros(n_features),
        }
    }
}

impl<B> LSTDLambda<B> {
    pub fn solve(&mut self) {
        let theta = self
            .a
            .solve(&self.b)
            .or_else(|_| pinv(&self.a).map(|ainv| ainv.dot(&self.b)));

        if let Ok(theta) = theta {
            self.theta.assign(&theta)
        }
    }
}

impl<'m, S, A, B> Handler<&'m Batch<S, A>> for LSTDLambda<B>
where B: Basis<&'m S, Value = Features>
{
    type Response = ();
    type Error = crate::fa::linear::Error;

    fn handle(&mut self, batch: &'m Batch<S, A>) -> Result<(), Self::Error> {
        for t in batch.iter().rev() {
            let (s, ns) = t.states();

            let phi_s = self.basis.project(s)?.into_dense();

            // Update trace:
            let c = self.lambda * self.gamma;

            self.z.zip_mut_with(&phi_s, move |x, &y| *x = c * *x + y);

            // Update matrices:
            self.b.scaled_add(t.reward, &self.z);

            if t.terminated() {
                self.a += &self
                    .z
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&phi_s.insert_axis(Axis(0)));
                self.z.fill(0.0);
            } else {
                let mut pd = self.basis.project(ns)?.into_dense();

                pd.zip_mut_with(&phi_s, |x, &y| *x = y - self.gamma * *x);

                self.a += &self
                    .z
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&pd.insert_axis(Axis(0)));
            }
        }

        self.solve();

        Ok(())
    }
}
