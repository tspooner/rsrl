use agents::{Predictor, TDPredictor};
use domains::Transition;
use fa::{Approximator, VFunction};
use std::marker::PhantomData;
use {Handler, Parameter};

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

impl<S, V: VFunction<S>> Handler<Transition<S, ()>> for TD<S, V> {
    fn handle_sample(&mut self, sample: &Transition<S, ()>) {
        let td_error = self.compute_td_error(sample);

        self.handle_td_error(&sample, td_error);
    }

    fn handle_terminal(&mut self, _: &Transition<S, ()>) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, V: VFunction<S>> Predictor<S> for TD<S, V> {
    fn predict(&mut self, s: &S) -> f64 { self.v_func.evaluate(s).unwrap() }
}

impl<S, V: VFunction<S>> TDPredictor<S> for TD<S, V> {
    fn handle_td_error(&mut self, sample: &Transition<S, ()>, error: f64) {
        let _ = self.v_func.update(&sample.from.state(), self.alpha * error);
    }

    fn compute_td_error(&self, sample: &Transition<S, ()>) -> f64 {
        let v = self.v_func.evaluate(&sample.from.state()).unwrap();
        let nv = self.v_func.evaluate(&sample.to.state()).unwrap();

        sample.reward + self.gamma * nv - v
    }
}
