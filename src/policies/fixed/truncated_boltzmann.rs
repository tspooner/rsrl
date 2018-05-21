use domains::Transition;
use geometry::Vector;
use rand::{thread_rng, ThreadRng};
use std::f64;
use super::{sample_probs, Policy, FinitePolicy, QPolicy};
use {Handler, Parameter};

pub struct TruncatedBoltzmann {
    c: Parameter,
    rng: ThreadRng,
}

impl TruncatedBoltzmann {
    pub fn new<T: Into<Parameter>>(c: T) -> Self {
        TruncatedBoltzmann {
            c: c.into(),
            rng: thread_rng(),
        }
    }

    fn kappa(c: f64, x: f64) -> f64 { c / (1.0 + (-x).exp()) }
}

impl<S, A> Handler<Transition<S, A>> for TruncatedBoltzmann {
    fn handle_terminal(&mut self, _: &Transition<S, A>) { self.c = self.c.step(); }
}

impl<S> Policy<S, usize> for TruncatedBoltzmann {
    fn sample(&mut self, s: &S) -> usize {
        let ps = self.probabilities(s);

        sample_probs(&mut self.rng, ps.as_slice().unwrap())
    }

    fn probability(&mut self, s: &S, a: usize) -> f64 {
        self.probabilities(s)[a]
    }
}

impl<S> FinitePolicy<S> for TruncatedBoltzmann {
    fn probabilities(&mut self, _: &S) -> Vector<f64> {
        unimplemented!()
    }
}

impl<S> QPolicy<S> for TruncatedBoltzmann {
    fn sample_qs(&mut self, s: &S, q_values: &[f64]) -> usize {
        let ps = self.probabilities_qs(s, q_values);

        sample_probs(&mut self.rng, ps.as_slice().unwrap())
    }

    fn probabilities_qs(&mut self, _: &S, q_values: &[f64]) -> Vector<f64> {
        let c = self.c.value();

        let mut z = 0.0;
        let ws: Vec<f64> = q_values.into_iter()
            .map(|q| {
                let v = TruncatedBoltzmann::kappa(c, *q).exp();
                z += v;

                v
            })
            .collect();

        ws.iter().map(|w| w / z).collect()
    }
}
