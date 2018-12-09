use core::*;
use domains::Transition;
use fa::{Approximator, VFunction, Parameterised, Projector, Projection, SimpleLFA};
use geometry::Space;
use ndarray::Axis;
use utils::argmaxima;

#[allow(non_camel_case_types)]
pub struct iLSTD<S, P: Projector<S>> {
    pub fa_theta: Shared<SimpleLFA<S, P>>,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub n_updates: usize,

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
            fa_theta,

            alpha: alpha.into(),
            gamma: gamma.into(),
            n_updates,

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

    fn solve(&mut self) {
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

impl<S, P: Projector<S>> Algorithm for iLSTD<S, P> {
    fn step_hyperparams(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, P: Projector<S>> OnlineLearner<S, A> for iLSTD<S, P> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let (s, ns) = (t.from.state(), t.to.state());

        // (D x 1)
        let phi_s = self.compute_dense_fv(s);
        let phi_ns = self.compute_dense_fv(ns);

        // (1 x D)
        let pd = if t.terminated() {
            phi_s.clone().insert_axis(Axis(0))
        } else {
            (phi_s.clone() - self.gamma.value()*phi_ns).insert_axis(Axis(0))
        };

        // (D x D)
        let delta_a = phi_s.clone().insert_axis(Axis(1)).dot(&pd);

        self.a += &delta_a;

        self.mu.scaled_add(t.reward, &phi_s);
        self.mu -= &delta_a.dot(&self.fa_theta.borrow().approximator.weights);

        self.solve();
    }
}

impl<S, P: Projector<S>> ValuePredictor<S> for iLSTD<S, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }
}

impl<S, A, P: Projector<S>> ActionValuePredictor<S, A> for iLSTD<S, P> {}

impl<S, P: Projector<S>> Parameterised for iLSTD<S, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
