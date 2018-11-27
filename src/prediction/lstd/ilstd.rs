use core::{Algorithm, Predictor, Parameter, Trace, Shared};
use domains::Transition;
use fa::{Approximator, VFunction, Parameterised, Projector, Projection, SimpleLFA};
use geometry::{Space, Vector, Matrix};
use ndarray::Axis;
use utils::argmaxima;

#[allow(non_camel_case_types)]
pub struct iLSTD<S, P: Projector<S>> {
    pub fa_theta: Shared<SimpleLFA<S, P>>,
    pub n_updates: usize,

    pub alpha: Parameter,
    pub gamma: Parameter,

    a: Matrix<f64>,
    mu: Vector<f64>,
}

impl<S, P: Projector<S>> iLSTD<S, P> {
    pub fn new<T1, T2>(fa_theta: Shared<SimpleLFA<S, P>>,
                       n_updates: usize,
                       alpha: T1, gamma: T2) -> Self
        where T1: Into<Parameter>,
              T2: Into<Parameter>
    {
        let n_features = fa_theta.borrow().projector.dim();

        iLSTD {
            fa_theta, n_updates,

            alpha: alpha.into(),
            gamma: gamma.into(),

            a: Matrix::zeros((n_features, n_features)),
            mu: Vector::zeros((n_features,)),
        }
    }
}

impl<S, P: Projector<S>> iLSTD<S, P> {
    #[inline(always)]
    fn compute_dense_fv(&self, s: &S) -> Vector<f64> {
        self.fa_theta.borrow().projector.project(s).expanded(self.a.rows())
    }

    #[inline(always)]
    fn do_update(&mut self, phi_s: Vector<f64>, pd: Vector<f64>, reward: f64) {
        // (D x D)
        let delta_a = phi_s.clone().insert_axis(Axis(1)).dot(&(pd.insert_axis(Axis(0))));

        self.a += &delta_a;

        self.mu.scaled_add(reward, &phi_s);
        self.mu -= &delta_a.dot(&self.fa_theta.borrow().approximator.weights);
    }

    fn consolidate(&mut self) {
        let mut fa = self.fa_theta.borrow_mut();
        let alpha = self.alpha.value();

        for _ in 0..self.n_updates {
            let (_, idx) = argmaxima(self.mu.mapv(|v| v.abs()).as_slice().unwrap());

            for j in idx {
                unsafe {
                    let update = alpha * self.mu.uget(j);

                    *fa.approximator.weights.uget_mut(j) += update;
                    self.mu.scaled_add(-update, &self.a.column(j));
                }
            }
        }
    }
}

impl<S, A, P: Projector<S>> Algorithm<S, A> for iLSTD<S, P> {
    fn handle_sample(&mut self, t: &Transition<S, A>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.compute_dense_fv(s);
        let phi_ns = self.compute_dense_fv(ns);

        let pd = phi_s.clone() - self.gamma.value()*phi_ns;

        self.do_update(phi_s, pd, t.reward);
        self.consolidate();
    }

    fn handle_terminal(&mut self, t: &Transition<S, A>) {
        {
            let phi_s = self.compute_dense_fv(t.from.state());

            self.do_update(phi_s.clone(), phi_s, t.reward);
            self.consolidate();
        }

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, P: Projector<S>> Predictor<S, A> for iLSTD<S, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }
}

impl<S, P: Projector<S>> Parameterised for iLSTD<S, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
