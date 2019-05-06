use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, Parameterised, Features, VFunction};
use crate::geometry::{Matrix, MatrixView, MatrixViewMut};

#[derive(Parameterised)]
pub struct TDLambda<F> {
    #[weights] pub fa_theta: F,

    pub alpha: Parameter,
    pub gamma: Parameter,

    trace: Trace,
}

impl<F> TDLambda<F> {
    pub fn new<T1, T2>(
        fa_theta: F,
        trace: Trace,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TDLambda {
            fa_theta,

            alpha: alpha.into(),
            gamma: gamma.into(),

            trace,
        }
    }
}

impl<F> Algorithm for TDLambda<F> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, F: VFunction<S>> OnlineLearner<S, A> for TDLambda<F> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let phi_s = self.fa_theta.embed(t.from.state());
        let v = self.fa_theta.evaluate(&phi_s).unwrap();

        let decay_rate = self.trace.lambda.value() * self.gamma.value();

        self.trace.decay(decay_rate);
        self.trace.update(&phi_s.expanded(self.fa_theta.n_features()));

        let z = self.trace.get();
        let td_error = if t.terminated() {
            self.trace.decay(0.0);

            t.reward - v
        } else {
            t.reward + self.gamma * self.predict_v(t.to.state()) - v
        };

        self.fa_theta.update(&Features::Dense(z), self.alpha * td_error).ok();
    }
}

impl<S, F: VFunction<S>> ValuePredictor<S> for TDLambda<F> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(&self.fa_theta.embed(s)).unwrap()
    }
}

impl<S, A, F: VFunction<S>> ActionValuePredictor<S, A> for TDLambda<F> {}
