use core::*;
use domains::Transition;
use fa::{Approximator, Parameterised, Projection, Projector, SimpleLFA, VFunction};
use geometry::Matrix;

pub struct TDLambda<S, P: Projector<S>> {
    pub fa_theta: Shared<SimpleLFA<S, P>>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    trace: Trace,
}

impl<S, P: Projector<S>> TDLambda<S, P> {
    pub fn new<T1, T2>(
        fa_theta: Shared<SimpleLFA<S, P>>,
        trace: Trace,
        alpha: T1,
        gamma: T2
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

impl<S, M: Projector<S>> Algorithm for TDLambda<S, M> {
    fn step_hyperparams(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, M: Projector<S>> OnlineLearner<S, A> for TDLambda<S, M> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let phi_s = self.fa_theta.borrow().projector.project(t.from.state());
        let v = self.fa_theta.borrow().evaluate_phi(&phi_s);

        let decay_rate = self.trace.lambda.value() * self.gamma.value();

        self.trace.decay(decay_rate);
        self.trace.update(&phi_s.expanded(self.fa_theta.borrow().projector.dim()));

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

impl<S, P: Projector<S>> ValuePredictor<S> for TDLambda<S, P> {
    fn predict_v(&mut self, s: &S) -> f64 {
        self.fa_theta.borrow().evaluate(s).unwrap()
    }
}

impl<S, A, P: Projector<S>> ActionValuePredictor<S, A> for TDLambda<S, P> {}

impl<S, P: Projector<S>> Parameterised for TDLambda<S, P> {
    fn weights(&self) -> Matrix<f64> {
        self.fa_theta.borrow().weights()
    }
}
