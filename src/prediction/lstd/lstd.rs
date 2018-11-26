use core::{Algorithm, Predictor, Parameter, Trace, Shared};
use domains::Transition;
use fa::{Approximator, VFunction, Parameterised, Projector, Projection, SimpleLFA};
use geometry::{Space, Vector, Matrix};
use ndarray::Axis;
use ndarray_linalg::solve::Solve;
use std::marker::PhantomData;
use utils::{argmaxima, pinv};


pub struct LSTD<S, P: Projector<S>> {
    pub fa_theta: Shared<SimpleLFA<S, P>>,
    pub gamma: Parameter,

    a: Matrix<f64>,
    b: Vector<f64>,

    phantom: PhantomData<S>,
}

impl<S, P: Projector<S>> LSTD<S, P> {
    pub fn new<T: Into<Parameter>>(fa_theta: Shared<SimpleLFA<S, P>>, gamma: T) -> Self {
        let n_features = fa_theta.borrow().projector.dim();

        LSTD {
            fa_theta,
            gamma: gamma.into(),

            a: Matrix::zeros((n_features, n_features)),
            b: Vector::zeros((n_features,)),

            phantom: PhantomData,
        }
    }
}

impl<S, P: Projector<S>> LSTD<S, P> {
    #[inline(always)]
    fn compute_dense_fv(&self, s: &S) -> Vector<f64> {
        let dim = self.a.rows();

        self.fa_theta.borrow().projector.project(s).expanded(dim)
    }

    #[inline(always)]
    fn do_update(&mut self, phi_s: Vector<f64>, pd: Vector<f64>, reward: f64) {
        self.b.scaled_add(reward, &phi_s);
        self.a += &phi_s.insert_axis(Axis(1)).dot(&(pd.insert_axis(Axis(0))));
    }

    pub fn solve(&mut self) {
        // First try the clean approach:
        if let Ok(theta) = self.a.solve(&self.b) {
            self.fa_theta.borrow_mut().approximator.weights.assign(&theta);

        // Otherwise solve via SVD:
        } else if let Ok(ainv) = pinv(&self.a) {
            let theta = ainv.dot(&self.b);

            self.fa_theta.borrow_mut().approximator.weights.assign(&theta);
        }
    }
}

impl<S, A, P: Projector<S>> Algorithm<S, A> for LSTD<S, P> {
    fn handle_sample(&mut self, t: &Transition<S, A>) {
        let (s, ns) = (t.from.state(), t.to.state());

        // (D x 1)
        let phi_s = self.compute_dense_fv(s);
        let phi_ns = self.compute_dense_fv(ns);

        // (1 x D)
        let pd = phi_s.clone() - self.gamma.value()*phi_ns;

        self.do_update(phi_s, pd, t.reward);
    }

    fn handle_terminal(&mut self, t: &Transition<S, A>) {
        {
            let s = t.from.state();
            let phi_s = self.compute_dense_fv(s);

            self.do_update(phi_s.clone(), phi_s, t.reward);
            self.solve();
        }

        self.gamma = self.gamma.step();
    }
}

impl<S, A, P: Projector<S>> Predictor<S, A> for LSTD<S, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }
}

impl<S, P: Projector<S>> Parameterised for LSTD<S, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
