use core::{Algorithm, Predictor, Parameter};
use domains::Transition;
use fa::VFunction;
use std::marker::PhantomData;

pub struct TD<S: ?Sized, V: VFunction<S>> {
    pub v_func: V,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S: ?Sized, V: VFunction<S>> TD<S, V> {
    pub fn new<T1, T2>(v_func: V, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TD {
            v_func: v_func,

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }
}

impl<S, A, V: VFunction<S>> Algorithm<S, A> for TD<S, V> {
    fn handle_sample(&mut self, sample: &Transition<S, A>) {
        let v = self.predict_v(&sample.from.state());
        let nv = self.predict_v(&sample.to.state());

        let td_error = sample.reward + self.gamma * nv - v;

        self.v_func.update(&sample.from.state(), self.alpha * td_error);
    }

    fn handle_terminal(&mut self, _: &Transition<S, A>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, V: VFunction<S>> Predictor<S, ()> for TD<S, V> {
    fn predict_v(&mut self, s: &S) -> f64 { self.v_func.evaluate(s).unwrap() }
}
