use core::{Algorithm, Predictor, Parameter, Trace, Shared};
use domains::Transition;
use fa::{Approximator, VFunction, Parameterised, Projector, Projection, SimpleLFA};
use geometry::{Space, Vector, Matrix};
use ndarray::Axis;
use ndarray_linalg::solve::Solve;
use std::marker::PhantomData;
use utils::{argmaxima, pinv};

pub struct LSTDLambda<S, P: Projector<S>> {
    pub fa_theta: Shared<SimpleLFA<S, P>>,
    pub gamma: Parameter,

    a: Matrix<f64>,
    b: Vector<f64>,

    trace: Trace,

    phantom: PhantomData<S>,
}

impl<S, P: Projector<S>> LSTDLambda<S, P> {
    pub fn new<T: Into<Parameter>>(trace: Trace,
                                   fa_theta: Shared<SimpleLFA<S, P>>,
                                   gamma: T) -> Self {
        let n_features = fa_theta.borrow().projector.dim();

        LSTDLambda {
            fa_theta,
            gamma: gamma.into(),

            a: Matrix::zeros((n_features, n_features)),
            b: Vector::zeros((n_features,)),

            trace,

            phantom: PhantomData,
        }
    }
}

impl<S, P: Projector<S>> LSTDLambda<S, P> {
    #[inline(always)]
    fn compute_dense_fv(&self, s: &S) -> Vector<f64> {
        self.fa_theta.borrow().projector.project(s).expanded(self.a.rows())
    }

    #[inline]
    fn update_trace(&mut self, phi: &Vector<f64>) -> Vector<f64> {
        let decay_rate = self.trace.lambda.value() * self.gamma.value();

        self.trace.decay(decay_rate);
        self.trace.update(phi);

        self.trace.get()
    }

    #[inline(always)]
    fn update_matrices(&mut self, z: Vector<f64>, pd: Vector<f64>, reward: f64) {
        self.b.scaled_add(reward, &z);
        self.a += &z.insert_axis(Axis(1)).dot(&(pd.insert_axis(Axis(0))));
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

impl<S, A, P: Projector<S>> Algorithm<S, A> for LSTDLambda<S, P> {
    fn handle_sample(&mut self, t: &Transition<S, A>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.compute_dense_fv(s);
        let phi_ns = self.compute_dense_fv(ns);

        let z = self.update_trace(&phi_s);
        let pd = phi_s - self.gamma.value()*phi_ns;

        self.update_matrices(z, pd, t.reward);
    }

    fn handle_terminal(&mut self, t: &Transition<S, A>) {
        {
            let phi_s = self.compute_dense_fv(t.from.state());

            let z = self.update_trace(&phi_s);

            self.update_matrices(z.clone(), phi_s.clone(), t.reward);
            self.solve();

            self.trace.decay(0.0);
        }

        self.gamma = self.gamma.step();
    }
}

impl<S, A, P: Projector<S>> Predictor<S, A> for LSTDLambda<S, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }
}

impl<S, P: Projector<S>> Parameterised for LSTDLambda<S, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
