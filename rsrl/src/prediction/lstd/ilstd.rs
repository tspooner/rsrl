use crate::{
    domains::Transition,
    fa::linear::{basis::Basis, Features},
    utils::argmaxima,
    Handler,
    Parameterised,
};
use ndarray::{Array1, Array2, Axis};

#[allow(non_camel_case_types)]
#[derive(Debug, Parameterised)]
pub struct iLSTD<B> {
    pub basis: B,
    #[weights]
    pub theta: Array1<f64>,

    pub alpha: f64,
    pub gamma: f64,
    pub n_updates: usize,

    a: Array2<f64>,
    mu: Array1<f64>,
}

impl<B: spaces::Space> iLSTD<B> {
    pub fn new(basis: B, alpha: f64, gamma: f64, n_updates: usize) -> Self {
        let n_features: usize = basis.dim().into();

        iLSTD {
            basis,
            theta: Array1::zeros(n_features),

            alpha,
            gamma,
            n_updates,

            a: Array2::eye(n_features),
            mu: Array1::zeros(n_features),
        }
    }

    pub fn to_lfa<O>(&self, optimiser: O) -> crate::fa::linear::ScalarLFA<B, O>
    where B: Clone,
    {
        crate::fa::linear::LFA {
            basis: self.basis.clone(),
            weights: self.theta.clone(),
            optimiser,
        }
    }

    pub fn into_lfa<O>(self, optimiser: O) -> crate::fa::linear::ScalarLFA<B, O> {
        crate::fa::linear::LFA {
            basis: self.basis,
            weights: self.theta,
            optimiser,
        }
    }
}

impl<B> iLSTD<B> {
    fn solve(&mut self) {
        for _ in 0..self.n_updates {
            let (idx, _) = argmaxima(self.mu.mapv(|v| v.abs()).iter().copied());

            for j in idx {
                unsafe {
                    let update = self.alpha * self.mu.uget(j);

                    *self.theta.uget_mut(j) += update;
                    self.mu.scaled_add(-update, &self.a.column(j));
                }
            }
        }
    }
}

impl<'m, S, A, B> Handler<&'m Transition<S, A>> for iLSTD<B>
where B: Basis<&'m S, Value = Features>
{
    type Response = ();
    type Error = crate::fa::linear::Error;

    fn handle(&mut self, t: &'m Transition<S, A>) -> Result<(), Self::Error> {
        let (s, ns) = t.states();

        // (D x 1)
        let phi_s = self.basis.project(s)?.into_dense();

        self.mu.scaled_add(t.reward, &phi_s);

        if t.terminated() {
            let phi_s = phi_s.insert_axis(Axis(1));

            // (D x D)
            let delta_a = phi_s.dot(&phi_s.t());

            self.a += &delta_a;
            self.mu -= &delta_a.dot(&self.theta);
        } else {
            let phi_ns = self.basis.project(ns)?.into_dense();

            // (1 x D)
            let pd = ((-self.gamma * phi_ns) + &phi_s).insert_axis(Axis(0));

            // (D x D)
            let delta_a = phi_s.view().insert_axis(Axis(1)).dot(&pd);

            self.a += &delta_a;
            self.mu -= &delta_a.dot(&self.theta);
        };

        self.solve();

        Ok(())
    }
}
