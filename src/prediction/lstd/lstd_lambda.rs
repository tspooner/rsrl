use crate::{
    core::*,
    domains::Transition,
    fa::{Approximator, VFunction, Parameterised, Projector, Projection, ScalarLFA},
    geometry::Space,
    utils::{argmaxima, pinv},
};
use ndarray::Axis;
use ndarray_linalg::solve::Solve;

pub struct LSTDLambda<M> {
    pub fa_theta: Shared<ScalarLFA<M>>,
    pub gamma: Parameter,

    trace: Trace,

    a: Matrix<f64>,
    b: Vector<f64>,
}

impl<M: Space> LSTDLambda<M> {
    pub fn new<T: Into<Parameter>>(fa_theta: Shared<ScalarLFA<M>>,
                                   trace: Trace, gamma: T) -> Self {
        let n_features = fa_theta.projector.dim();

        LSTDLambda {
            fa_theta,

            gamma: gamma.into(),

            trace,

            a: Matrix::zeros((n_features, n_features)),
            b: Vector::zeros((n_features,)),
        }
    }
}

impl<M> LSTDLambda<M> {
    #[inline]
    fn update_trace(&mut self, phi: &Vector<f64>) -> Vector<f64> {
        let decay_rate = self.trace.lambda.value() * self.gamma.value();

        self.trace.decay(decay_rate);
        self.trace.update(phi);

        self.trace.get()
    }

    pub fn solve(&mut self) {
        // First try the clean approach:
        if let Ok(theta) = self.a.solve(&self.b) {
            self.fa_theta.borrow_mut().evaluator.weights.assign(&theta);

        // Otherwise solve via SVD:
        } else if let Ok(ainv) = pinv(&self.a) {
            let theta = ainv.dot(&self.b);

            self.fa_theta.borrow_mut().evaluator.weights.assign(&theta);
        }
    }
}

impl<M> Algorithm for LSTDLambda<M> {
    fn handle_terminal(&mut self) {
        self.gamma = self.gamma.step();
    }
}

impl<S, A, M: Projector<S>> BatchLearner<S, A> for LSTDLambda<M> {
    fn handle_batch(&mut self, ts: &[Transition<S, A>]) {
        ts.into_iter().for_each(|t| {
            let phi_s = self.fa_theta.projector
                .project(t.from.state())
                .expanded(self.a.rows());
            let z = self.update_trace(&phi_s);

            self.b.scaled_add(t.reward, &z);

            let pd = if t.terminated() {
                self.trace.decay(0.0);

                phi_s
            } else {
                let phi_ns = self.fa_theta.projector
                    .project(t.to.state())
                    .expanded(self.a.rows());

                phi_s - self.gamma.value()*phi_ns
            }.insert_axis(Axis(0));

            self.a += &z.insert_axis(Axis(1)).dot(&pd);
        });

        self.solve();
    }
}

impl<S, M> ValuePredictor<S> for LSTDLambda<M>
where
    ScalarLFA<M>: VFunction<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(s).unwrap()
    }
}

impl<S, A, M> ActionValuePredictor<S, A> for LSTDLambda<M>
where
    ScalarLFA<M>: VFunction<S>,
{}

impl<M> Parameterised for LSTDLambda<M>
where
    ScalarLFA<M>: Parameterised,
{
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }
}
