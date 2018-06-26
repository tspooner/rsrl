use super::{sample_probs, FinitePolicy, Policy};
use core::Parameter;
use domains::Transition;
use fa::SharedQFunction;
use geometry::Vector;
use rand::{thread_rng, ThreadRng};
use std::f64;

fn kappa(c: f64, x: f64) -> f64 { c / (1.0 + (-x).exp()) }

pub struct TruncatedBoltzmann<S> {
    q_func: SharedQFunction<S>,

    c: Parameter,
    rng: ThreadRng,
}

impl<S> TruncatedBoltzmann<S> {
    pub fn new<T: Into<Parameter>>(q_func: SharedQFunction<S>, c: T) -> Self {
        TruncatedBoltzmann {
            q_func,

            c: c.into(),
            rng: thread_rng(),
        }
    }
}

impl<S> Policy<S> for TruncatedBoltzmann<S> {
    type Action = usize;

    fn sample(&mut self, s: &S) -> usize {
        let ps = self.probabilities(s);

        sample_probs(&mut self.rng, ps.as_slice().unwrap())
    }

    fn probability(&mut self, s: &S, a: usize) -> f64 { self.probabilities(s)[a] }

    fn handle_terminal(&mut self, _: &Transition<S, usize>) { self.c = self.c.step(); }
}

impl<S> FinitePolicy<S> for TruncatedBoltzmann<S> {
    fn probabilities(&mut self, s: &S) -> Vector<f64> {
        let c = self.c.value();

        let mut z = 0.0;
        let ws: Vec<f64> = self
            .q_func
            .borrow()
            .evaluate(s)
            .unwrap()
            .into_iter()
            .map(|q| {
                let v = kappa(c, *q).exp();
                z += v;

                v
            })
            .collect();

        ws.iter().map(|w| w / z).collect()
    }
}
