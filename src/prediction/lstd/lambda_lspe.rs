use core::*;
use domains::Transition;
use fa::{Approximator, VFunction, Parameterised, Projector, Projection, SimpleLFA};
use geometry::Space;
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

    fn solve(&mut self) {
        // First try the clean approach otherwise solve via SVD:
        if let Ok(theta) = self.a.solve(&self.b).or_else(|_| {
            pinv(&self.a).map(|ainv| ainv.dot(&self.b))
        }) {
            self.fa_theta.borrow_mut().approximator.weights *= 1.0 - self.alpha;
            self.fa_theta.borrow_mut().approximator.weights.scaled_add(self.alpha.value(), &theta);

            self.a.fill(0.0);
            self.b.fill(0.0);
            self.delta = 0.0;
        }
    }
}

impl<S, P: Projector<S>> Algorithm for LambdaLSPE<S, P> {
    fn step_hyperparams(&mut self) {
        self.gamma = self.gamma.step();
    }
}

impl<S, A, P: Projector<S>> BatchLearner<S, A> for LambdaLSPE<S, P> {
    fn handle_batch(&mut self, batch: &[Transition<S, A>]) {
        batch.into_iter().rev().for_each(|ref t| {
            let phi_s = self.compute_dense_fv(t.from.state());
            let phi_ns = self.compute_dense_fv(t.to.state());

            let v = self.compute_v(&phi_s);

            if t.terminated() {
                let residual = t.reward - v;

                self.b.scaled_add(v + self.gamma * self.lambda * self.delta + residual, &phi_s);
                self.a += &phi_s.clone().insert_axis(Axis(1)).dot(&(phi_s.insert_axis(Axis(0))));

            } else {
                let residual = t.reward + self.gamma * self.compute_v(&phi_ns) - v;

                self.delta = self.gamma * self.lambda * self.delta + residual;

                self.b.scaled_add(v + self.delta, &phi_s);
                self.a += &phi_s.clone().insert_axis(Axis(1)).dot(&(phi_s.insert_axis(Axis(0))));
            };
        });

        self.solve();
    }
}

impl<S, P: Projector<S>> ValuePredictor<S> for LambdaLSPE<S, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }
}

impl<S, A, P: Projector<S>> ActionValuePredictor<S, A> for LambdaLSPE<S, P> {}

impl<S, P: Projector<S>> Parameterised for LambdaLSPE<S, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
