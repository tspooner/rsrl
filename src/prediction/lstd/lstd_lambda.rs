use crate::{
    core::*,
    domains::Transition,
    fa::{Approximator, VFunction, Parameterised},
    geometry::{Space, Matrix, MatrixView, MatrixViewMut},
    utils::{argmaxima, pinv},
};
use ndarray::Axis;
use ndarray_linalg::solve::Solve;

#[derive(Parameterised)]
pub struct LSTDLambda<F> {
    #[weights] pub fa_theta: F,
    pub gamma: Parameter,

    trace: Trace,

    a: Matrix<f64>,
    b: Vector<f64>,
}

impl<F: Parameterised> LSTDLambda<F> {
    pub fn new<T: Into<Parameter>>(fa_theta: F, trace: Trace, gamma: T) -> Self {
        let dim = fa_theta.weights_dim();

        LSTDLambda {
            fa_theta,

            gamma: gamma.into(),

            trace,

            a: Matrix::zeros(dim),
            b: Vector::zeros(dim.0),
        }
    }
}

impl<F> LSTDLambda<F> {
    #[inline]
    fn update_trace(&mut self, phi: &Vector<f64>) -> Vector<f64> {
        let decay_rate = self.trace.lambda.value() * self.gamma.value();

        self.trace.decay(decay_rate);
        self.trace.update(phi);

        self.trace.get()
    }
}

impl<F: Parameterised> LSTDLambda<F> {
    pub fn solve(&mut self) {
        let mut w = self.fa_theta.weights_view_mut();

        if let Ok(theta) = self.a.solve(&self.b) {
            // First try the clean approach:
            w.assign(&theta);
        } else if let Ok(ainv) = pinv(&self.a) {
            // Otherwise solve via SVD:
            w.assign(&ainv.dot(&self.b));
        }
    }
}

impl<F> Algorithm for LSTDLambda<F> {
    fn handle_terminal(&mut self) {
        self.gamma = self.gamma.step();
    }
}

impl<S, A, F: VFunction<S> + Parameterised> BatchLearner<S, A> for LSTDLambda<F> {
    fn handle_batch(&mut self, ts: &[Transition<S, A>]) {
        ts.into_iter().for_each(|t| {
            let phi_s = self.fa_theta
                .embed(t.from.state())
                .expanded(self.a.rows());
            let z = self.update_trace(&phi_s);

            self.b.scaled_add(t.reward, &z);

            let pd = if t.terminated() {
                self.trace.decay(0.0);

                phi_s
            } else {
                let phi_ns = self.fa_theta
                    .embed(t.to.state())
                    .expanded(self.a.rows());

                phi_s - self.gamma.value()*phi_ns
            }.insert_axis(Axis(0));

            self.a += &z.insert_axis(Axis(1)).dot(&pd);
        });

        self.solve();
    }
}

impl<S, F: VFunction<S>> ValuePredictor<S> for LSTDLambda<F> {
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(&self.fa_theta.embed(s)).unwrap()
    }
}

impl<S, A, F: VFunction<S>> ActionValuePredictor<S, A> for LSTDLambda<F> {}
