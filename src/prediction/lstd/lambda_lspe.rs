use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, VFunction, Parameterised, Projector, Projection, ScalarLFA};
use crate::geometry::Space;
use ndarray::Axis;
use ndarray_linalg::solve::Solve;
use crate::utils::{argmaxima, pinv};

pub struct LambdaLSPE<M> {
    pub fa_theta: Shared<ScalarLFA<M>>,

    pub alpha: Parameter,
    pub gamma: Parameter,
    pub lambda: Parameter,

    a: Matrix<f64>,
    b: Vector<f64>,
    delta: f64,
}

impl<M: Space> LambdaLSPE<M> {
    pub fn new<T1, T2, T3>(fa_theta: Shared<ScalarLFA<M>>,
                           alpha: T1, gamma: T2, lambda: T3) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
        T3: Into<Parameter>,
    {
        let n_features = fa_theta.projector.dim();

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

impl<M> LambdaLSPE<M> {
    #[inline(always)]
    fn compute_v(&self, phi: &Vector<f64>) -> f64 {
        self.fa_theta.approximator.weights.dot(phi)
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

impl<M> Algorithm for LambdaLSPE<M> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
        self.lambda = self.lambda.step();
    }
}

impl<S, A, M: Projector<S>> BatchLearner<S, A> for LambdaLSPE<M> {
    fn handle_batch(&mut self, batch: &[Transition<S, A>]) {
        batch.into_iter().rev().for_each(|ref t| {
            let phi_s = self.fa_theta.projector
                .project(t.from.state())
                .expanded(self.a.rows());
            let v = self.compute_v(&phi_s);

            if t.terminated() {
                let residual = t.reward - v;

                self.b.scaled_add(v + self.gamma * self.lambda * self.delta + residual, &phi_s);
                self.a += &phi_s.clone().insert_axis(Axis(1)).dot(&(phi_s.insert_axis(Axis(0))));

            } else {
                let phi_ns = self.fa_theta.projector
                    .project(t.to.state())
                    .expanded(self.a.rows());
                let residual = t.reward + self.gamma * self.compute_v(&phi_ns) - v;

                self.delta = self.gamma * self.lambda * self.delta + residual;

                self.b.scaled_add(v + self.delta, &phi_s);
                self.a += &phi_s.clone().insert_axis(Axis(1)).dot(&(phi_s.insert_axis(Axis(0))));
            };
        });

        self.solve();
    }
}

impl<S, M> ValuePredictor<S> for LambdaLSPE<M>
where
    ScalarLFA<M>: VFunction<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(s).unwrap()
    }
}

impl<S, A, M> ActionValuePredictor<S, A> for LambdaLSPE<M>
where
    ScalarLFA<M>: VFunction<S>,
{}

impl<M> Parameterised for LambdaLSPE<M>
where
    ScalarLFA<M>: Parameterised,
{
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }
}
