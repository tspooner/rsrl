use agents::{Agent, Predictor};
use fa::VFunction;
use std::marker::PhantomData;
use Parameter;

pub struct EveryVisitMC<S, V: VFunction<S>> {
    pub v_func: V,
    observations: Vec<(S, f64)>,

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
            observations: vec![],

            alpha: alpha.into(),
            gamma: gamma.into(),

            phantom: PhantomData,
        }
    }

    pub fn propagate(&mut self) {
        let mut sum = 0.0;

        for (s, r) in self.observations.drain(0..).rev() {
            sum = r + self.gamma * sum;

            let v_est = self.v_func.evaluate(&s).unwrap();
            let _ = self.v_func.update(&s, self.alpha * (sum - v_est));
        }
    }
}

impl<S: Clone, V: VFunction<S>> Agent for EveryVisitMC<S, V> {
    type Sample = (S, S, f64);

    fn handle_sample(&mut self, sample: &Self::Sample) {
        self.observations.push((sample.0.clone(), sample.2.clone()));
    }

    fn handle_terminal(&mut self, _: &Self::Sample) {
        self.propagate();

        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S: Clone, V: VFunction<S>> Predictor<S> for EveryVisitMC<S, V> {
    fn evaluate(&self, s: &S) -> f64 { self.v_func.evaluate(s).unwrap() }
}
