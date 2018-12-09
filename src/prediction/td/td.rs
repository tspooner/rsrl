use core::*;
use domains::Transition;
use fa::{Parameterised, VFunction};
use geometry::Matrix;
use std::marker::PhantomData;

pub struct TD<S, V> {
    pub v_func: Shared<V>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, V: VFunction<S>> TD<S, V> {
    pub fn new<T1, T2>(v_func: Shared<V>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TD {
            v_func,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, V: VFunction<S>> Algorithm for TD<S, V> {
    fn step_hyperparams(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, A, V: VFunction<S>> OnlineLearner<S, A> for TD<S, V> {
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

impl<S, V: VFunction<S>> ValuePredictor<S> for TD<S, V> {
    fn predict_v(&mut self, s: &S) -> f64 { self.v_func.borrow().evaluate(s).unwrap() }
}

impl<S, A, V: VFunction<S>> ActionValuePredictor<S, A> for TD<S, V> {}

impl<S, V: Parameterised> Parameterised for TD<S, V> {
    fn weights(&self) -> Matrix<f64> {
        self.v_func.borrow().weights()
    }
}
