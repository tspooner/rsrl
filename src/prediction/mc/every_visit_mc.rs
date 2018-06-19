use core::{Algorithm, Predictor, Parameter};
use domains::Transition;
use fa::VFunction;
use std::marker::PhantomData;

pub struct EveryVisitMC<S, V: VFunction<S>> {
    pub v_func: V,
    pub cache: Vec<(S, f64)>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    phantom: PhantomData<S>,
}

impl<S, V: VFunction<S>> EveryVisitMC<S, V> {
    pub fn new<T1, T2>(v_func: V, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        EveryVisitMC {
            v_func: v_func,
            cache: vec![],

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }

    pub fn propagate(&mut self) {
        let mut sum = 0.0;

        for (s, r) in self.cache.drain(0..).rev() {
            sum = r + self.gamma * sum;

            let v_est = self.v_func.evaluate(&s).unwrap();
            let _ = self.v_func.update(&s, self.alpha * (sum - v_est));
        }
    }
}

impl<S: Clone, A, V: VFunction<S>> Algorithm<S, A> for EveryVisitMC<S, V> {
    fn handle_sample(&mut self, sample: &Transition<S, A>) {
        self.cache.push((sample.from.state().clone(), sample.reward));
    }

    fn handle_terminal(&mut self, _: &Transition<S, A>) {
        self.propagate();

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S: Clone, A, V: VFunction<S>> Predictor<S, A> for EveryVisitMC<S, V> {
    fn v(&mut self, s: &S) -> f64 { self.v_func.evaluate(s).unwrap() }
}
