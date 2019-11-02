use crate::{
    OnlineLearner,
    domains::Transition,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction,
        linear::LinearStateFunction,
    },
    prediction::ValuePredictor,
    utils::argmaxima,
};
use ndarray::{Array1, Array2, Axis};

#[allow(non_camel_case_types)]
#[derive(Parameterised)]
pub struct iLSTD<F> {
    #[weights] pub fa_theta: F,

    pub alpha: f64,
    pub gamma: f64,
    pub n_updates: usize,

    a: Array2<f64>,
    mu: Array1<f64>,
}

impl<F: Parameterised> iLSTD<F> {
    pub fn new(fa_theta: F, alpha: f64, gamma: f64, n_updates: usize) -> Self {
        let dim = fa_theta.weights_dim();

        iLSTD {
            fa_theta,

            alpha,
            gamma,
            n_updates,

            a: Array2::eye(dim[0]),
            mu: Array1::zeros(dim[0]),
        }
    }
}

impl<F: Parameterised> iLSTD<F> {
    fn solve(&mut self) {
        let mut w = self.fa_theta.weights_view_mut();

        for _ in 0..self.n_updates {
            let (_, idx) = argmaxima(self.mu.mapv(|v| v.abs()).as_slice().unwrap());

            for j in idx {
                unsafe {
                    let update = self.alpha * self.mu.uget(j);

                    *w.uget_mut((j, 0)) += update;
                    self.mu.scaled_add(-update, &self.a.column(j));
                }
            }
        }
    }
}

impl<S, A, F> OnlineLearner<S, A> for iLSTD<F>
where
    F: LinearStateFunction<S, Output = f64>
{
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let (s, ns) = t.states();

        // (D x 1)
        let phi_s = self.fa_theta.features(s).expanded();

        self.mu.scaled_add(t.reward, &phi_s);

        if t.terminated() {
            let phi_s = phi_s.insert_axis(Axis(1));

            // (D x D)
            let delta_a = phi_s.dot(&phi_s.t());

            self.a += &delta_a;
            self.mu -= &delta_a.dot(&self.fa_theta.weights_view()).column(0);
        } else {
            let phi_ns = self.fa_theta.features(ns).expanded();

            // (1 x D)
            let pd = ((-self.gamma * phi_ns) + &phi_s).insert_axis(Axis(0));

            // (D x D)
            let delta_a = phi_s.view().insert_axis(Axis(1)).dot(&pd);

            self.a += &delta_a;
            self.mu -= &delta_a.dot(&self.fa_theta.weights_view()).column(0);
        };

        self.solve();
    }
}

impl<S, F> ValuePredictor<S> for iLSTD<F>
where
    F: StateFunction<S, Output = f64>
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s)
    }
}
