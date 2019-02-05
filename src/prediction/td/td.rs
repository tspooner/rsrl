use crate::core::*;
use crate::domains::Transition;
use crate::fa::{Parameterised, VFunction};
use crate::geometry::Matrix;

pub struct TD<V> {
    pub v_func: Shared<V>,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<V> TD<V> {
    pub fn new<T1, T2>(v_func: Shared<V>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TD {
            v_func,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<V> Algorithm for TD<V> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, V: VFunction<S>> OnlineLearner<S, A> for TD<V> {
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let s = t.from.state();
        let v = self.predict_v(s);

        let td_error = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * self.predict_v(t.to.state()) - v
        };

        self.v_func.borrow_mut().update(s, self.alpha * td_error).ok();
    }
}

impl<S, V: VFunction<S>> ValuePredictor<S> for TD<V> {
    fn predict_v(&mut self, s: &S) -> f64 { self.v_func.evaluate(s).unwrap() }
}

impl<S, A, V: VFunction<S>> ActionValuePredictor<S, A> for TD<V> {}

impl<V: Parameterised> Parameterised for TD<V> {
    fn weights(&self) -> Matrix<f64> {
        self.v_func.weights()
    }
}
