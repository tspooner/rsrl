use core::{Algorithm, Predictor, Parameter, Trace, Shared};
use domains::Transition;
use fa::{Approximator, VFunction, Parameterised, Projector, Projection, SimpleLFA};
use geometry::{Space, Vector, Matrix};
use ndarray::Axis;
use ndarray_linalg::solve::Solve;
use utils::{argmaxima, pinv};

pub struct LambdaLSPE<S, P: Projector<S>> {
    pub fa_theta: Shared<SimpleLFA<S, P>>,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub lambda: Parameter,

    a: Matrix<f64>,
    b: Vector<f64>,
    delta: f64,
}

impl<S, P: Projector<S>> LambdaLSPE<S, P> {
    pub fn new<T1, T2, T3>(fa_theta: Shared<SimpleLFA<S, P>>, alpha: T1, gamma: T2,
                           lambda: T3) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        let n_features = fa_theta.borrow().projector.dim();

        LambdaLSPE {
            fa_theta,

            alpha: alpha.into(),
            gamma: gamma.into(),
            lambda: lambda.into(),

            a: Matrix::zeros((n_features, n_features)),
            b: Vector::zeros((n_features,)),
            delta: 0.0,
        }
    }
}

impl<S, P: Projector<S>> LambdaLSPE<S, P> {
    #[inline(always)]
    fn compute_dense_fv(&self, s: &S) -> Vector<f64> {
        self.fa_theta.borrow().projector.project(s).expanded(self.a.rows())
    }

    #[inline(always)]
    fn compute_v(&self, phi: &Vector<f64>) -> f64 {
        self.fa_theta.borrow().approximator.weights.dot(phi)
    }

    #[inline(always)]
    fn update_matrices(&mut self, phi_s: Vector<f64>, v: f64, td_error: f64) {
        self.delta = self.gamma * self.lambda * self.delta + td_error;

        self.b.scaled_add(v + self.delta, &phi_s);
        self.a += &phi_s.clone().insert_axis(Axis(1)).dot(&(phi_s.insert_axis(Axis(0))));
    }

    #[inline(always)]
    fn update_theta(&mut self, theta: Vector<f64>) {
        self.fa_theta.borrow_mut().approximator.weights *= 1.0 - self.alpha;
        self.fa_theta.borrow_mut().approximator.weights.scaled_add(self.alpha.value(), &theta);
    }

    pub fn solve(&mut self) {
        // First try the clean approach:
        if let Ok(theta) = self.a.solve(&self.b) {
            self.update_theta(theta);

        // Otherwise solve via SVD:
        } else if let Ok(ainv) = pinv(&self.a) {
            let theta = ainv.dot(&self.b);

            self.update_theta(theta);
        }

        self.a.fill(0.0);
        self.b.fill(0.0);
        self.delta = 0.0;
    }
}

impl<S, A, P: Projector<S>> Algorithm<S, A> for LambdaLSPE<S, P> {
    fn handle_sample(&mut self, t: &Transition<S, A>) {
        let phi_s = self.compute_dense_fv(t.from.state());
        let phi_ns = self.compute_dense_fv(t.to.state());

        let v = self.compute_v(&phi_s);
        let nv = self.compute_v(&phi_ns);

        let td_error = t.reward + self.gamma * nv - v;

        self.update_matrices(phi_s, v, td_error);
    }

    fn handle_terminal(&mut self, t: &Transition<S, A>) {
        {
            self.handle_sample(t);

            let phi_terminal = self.compute_dense_fv(t.to.state());
            let v_terminal = self.compute_v(&phi_terminal);
            let td_error = t.reward - v_terminal;

            self.update_matrices(phi_terminal, v_terminal, td_error);
            self.solve();
        }

        self.gamma = self.gamma.step();
    }
}

impl<S, A, P: Projector<S>> Predictor<S, A> for LambdaLSPE<S, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }
}

impl<S, P: Projector<S>> Parameterised for LambdaLSPE<S, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
