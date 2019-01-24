use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, VFunction, Parameterised, Projector, Projection, ScalarLFA};
use crate::geometry::Space;
use ndarray::Axis;
use crate::utils::argmaxima;

#[allow(non_camel_case_types)]
pub struct iLSTD<M> {
    pub fa_theta: Shared<ScalarLFA<M>>,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub n_updates: usize,

    a: Matrix<f64>,
    mu: Vector<f64>,
}

impl<M: Space> iLSTD<M> {
    pub fn new<T1, T2>(fa_theta: Shared<ScalarLFA<M>>,
                       n_updates: usize, alpha: T1, gamma: T2) -> Self
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

impl<M> iLSTD<M> {
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

impl<M> Algorithm for iLSTD<M> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, M: Projector<S>> OnlineLearner<S, A> for iLSTD<M> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        // (D x 1)
        let phi_s = self.fa_theta.borrow().projector
            .project(t.from.state())
            .expanded(self.a.rows());

        // (1 x D)
        let pd = if t.terminated() {
            phi_s.clone().insert_axis(Axis(0))
        } else {
            let phi_ns = self.fa_theta.borrow().projector
                .project(t.to.state())
                .expanded(self.a.rows());

            (phi_s.clone() - self.gamma.value() * phi_ns).insert_axis(Axis(0))
        };

        // (D x D)
        let delta_a = phi_s.clone().insert_axis(Axis(1)).dot(&pd);

        self.a += &delta_a;

        self.mu.scaled_add(t.reward, &phi_s);
        self.mu -= &delta_a.dot(&self.fa_theta.borrow().approximator.weights);

        self.solve();
    }
}

impl<S, M> ValuePredictor<S> for iLSTD<M>
where
    ScalarLFA<M>: VFunction<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }
}

impl<S, A, M> ActionValuePredictor<S, A> for iLSTD<M>
where
    ScalarLFA<M>: VFunction<S>,
{}

impl<M> Parameterised for iLSTD<M>
where
    ScalarLFA<M>: Parameterised,
{
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
