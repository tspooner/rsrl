use crate::{
    domains::Transition,
    fa::linear::{basis::Basis, Features},
    Handler,
};
use ndarray::{Array1, Array2, Axis};
use spaces::Space;

#[derive(Debug, Parameterised)]
pub struct RecursiveLSTD<B> {
    pub basis: B,
    #[weights]
    pub theta: Array1<f64>,

    pub gamma: f64,

    c_mat: Array2<f64>,
}

impl<B: Space> RecursiveLSTD<B> {
    pub fn new(basis: B, gamma: f64) -> Self {
        let n_features: usize = basis.dim().into();

        RecursiveLSTD {
            basis,
            theta: Array1::zeros(n_features),

            gamma,

            c_mat: Array2::eye(n_features) * 1e-5,
        }
    }
}

impl<'m, S, A, B> Handler<&'m Transition<S, A>> for RecursiveLSTD<B>
where B: Basis<&'m S, Value = Features>
{
    type Response = ();
    type Error = crate::fa::linear::Error;

    fn handle(&mut self, t: &'m Transition<S, A>) -> Result<(), Self::Error> {
        let (s, ns) = t.states();

        let phi_s = self.basis.project(s)?;
        let theta_s = phi_s.dot(&self.theta);

        if t.terminated() {
            // (D x 1)
            let phi_s = phi_s.into_dense().insert_axis(Axis(1));

            // (D x 1) <- (D x D) . (D x 1)
            let v = self.c_mat.dot(&phi_s);

            // (1 x D) <- (D x 1)T
            let g = v.t();

            // (1 x 1) <- (1 x D) . (D x 1)
            let a = 1.0 + unsafe { g.dot(&phi_s).uget((0, 0)) };
            let residual = t.reward - theta_s;

            self.c_mat.fill(0.0);
            self.theta.scaled_add(residual / a, &v);
        } else {
            let phi_ns = self.basis.project(ns)?;
            let theta_ns = phi_ns.dot(&self.theta);

            // (D x 1)
            let phi_s = phi_s.into_dense().insert_axis(Axis(1));
            let phi_ns = phi_ns.into_dense().insert_axis(Axis(1));

            let pd = (-self.gamma * phi_ns) + &phi_s;

            // (1 x D) <- ((D x D) . (D x 1))T
            //  Note: self.permuted_axes([1, 0]) is equivalent to taking the transpose.
            let g = self.c_mat.dot(&pd).permuted_axes([1, 0]);

            // (1 x 1) <- (1 x D) . (D x 1)
            let a = 1.0 + unsafe { g.dot(&phi_s).uget((0, 0)) };

            // (D x 1) <- (D x D) . (D x 1)
            let v = self.c_mat.dot(&phi_s);
            let residual = t.reward + self.gamma * theta_ns - theta_s;

            // (D x D) <- (D x 1) . (1 x D)
            let vg = v.dot(&g);

            self.c_mat.scaled_add(-1.0 / a, &vg);
            self.theta.scaled_add(residual / a, &v);
        }

        Ok(())
    }
}
