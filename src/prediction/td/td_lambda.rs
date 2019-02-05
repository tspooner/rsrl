use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Approximator, Parameterised, Projection, Projector, ScalarLFA, VFunction};
use crate::geometry::Matrix;

pub struct TDLambda<M> {
    pub fa_theta: Shared<ScalarLFA<M>>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    trace: Trace,
}

impl<M> TDLambda<M> {
    pub fn new<T1, T2>(
        fa_theta: Shared<ScalarLFA<M>>,
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

impl<M> Algorithm for TDLambda<M> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, M: Projector<S>> OnlineLearner<S, A> for TDLambda<M> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let phi_s = self.fa_theta.projector.project(t.from.state());
        let v = self.fa_theta.evaluate_phi(&phi_s);

        let decay_rate = self.trace.lambda.value() * self.gamma.value();

        self.trace.decay(decay_rate);
        self.trace.update(&phi_s.expanded(self.fa_theta.projector.dim()));

        let z = self.trace.get();
        let td_error = if t.terminated() {
            self.trace.decay(0.0);

            t.reward - v
        } else {
            t.reward + self.gamma * self.predict_v(t.to.state()) - v
        };

        self.fa_theta.borrow_mut().update_phi(&Projection::Dense(z), self.alpha * td_error);
    }
}

impl<S, M> ValuePredictor<S> for TDLambda<M>
where
    ScalarLFA<M>: VFunction<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.evaluate(s).unwrap()
    }
}

impl<S, A, M> ActionValuePredictor<S, A> for TDLambda<M>
where
    ScalarLFA<M>: VFunction<S>,
{}

impl<M> Parameterised for TDLambda<M>
where
    ScalarLFA<M>: Parameterised
{
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.weights()
    }
}
