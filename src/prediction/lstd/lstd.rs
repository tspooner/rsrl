use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, VFunction, Parameterised, Projector, Projection, ScalarLFA};
use crate::geometry::Space;
use ndarray::Axis;
use ndarray_linalg::solve::Solve;
use crate::utils::{argmaxima, pinv};

pub struct LSTD<M> {
    pub fa_theta: Shared<ScalarLFA<M>>,

    pub gamma: Parameter,

    a: Matrix<f64>,
    b: Vector<f64>,
}

impl<M: Space> LSTD<M> {
    pub fn new<T: Into<Parameter>>(fa_theta: Shared<ScalarLFA<M>>, gamma: T) -> Self {
        let n_features = fa_theta.projector.dim();

        LSTD {
            fa_theta,

            gamma: gamma.into(),

            a: Matrix::zeros((n_features, n_features)),
            b: Vector::zeros((n_features,)),
        }
    }
}

impl<M> Algorithm for LSTD<M> {
    fn handle_terminal(&mut self) {
        self.gamma = self.gamma.step();
    }
}

impl<S, A, M: Projector<S>> BatchLearner<S, A> for LSTD<M> {
    fn handle_batch(&mut self, ts: &[Transition<S, A>]) {
        ts.into_iter().for_each(|ref t| {
            let phi_s = self.fa_theta.projector
                .project(t.from.state())
                .expanded(self.a.rows());
            let pd = if t.terminated() {
                phi_s.clone()
            } else {
                let phi_ns = self.fa_theta.projector
                    .project(t.to.state())
                    .expanded(self.a.rows());

                phi_s.clone() - self.gamma.value()*phi_ns.clone()
            }.insert_axis(Axis(0));

            self.b.scaled_add(t.reward, &phi_s);
            self.a += &phi_s.insert_axis(Axis(1)).dot(&pd);
        });

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

impl<S, M> ValuePredictor<S> for LSTD<M>
where
    ScalarLFA<M>: VFunction<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(s).unwrap()
    }
}

impl<S, A, M> ActionValuePredictor<S, A> for LSTD<M>
where
    ScalarLFA<M>: VFunction<S>,
{}

impl<M> Parameterised for LSTD<M>
where
    ScalarLFA<M>: Parameterised
{
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }
}
