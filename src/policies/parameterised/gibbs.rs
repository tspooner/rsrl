use crate::core::*;
use crate::fa::{Approximator, MultiLFA, Parameterised, Projector};
use crate::policies::{DifferentiablePolicy, FinitePolicy, ParameterisedPolicy, Policy};
use rand::{rngs::ThreadRng, thread_rng, Rng};
use std::{f64, ops::AddAssign};

pub struct Gibbs<S, M: Projector<S>> {
    pub fa: MultiLFA<S, M>,

    rng: ThreadRng,
}

fn probabilities_from_values(values: &[f64]) -> Vector<f64> {
    let mut z = 0.0;
    let ws: Vec<f64> = values
        .iter()
        .map(|v| {
            let v = v.exp();
            z += v;

            v
        })
        .collect();

    ws.iter().map(|w| (w / z).min(1e50)).collect()
}

impl<S, M: Projector<S>> Gibbs<S, M> {
    pub fn new(fa: MultiLFA<S, M>) -> Self {
        Gibbs {
            fa,

            rng: thread_rng(),
        }
    }
}

impl<S, M: Projector<S>> Algorithm for Gibbs<S, M> {}

impl<S, M: Projector<S>> Policy<S> for Gibbs<S, M> {
    type Action = usize;

    fn sample(&mut self, input: &S) -> usize {
        let ps = self.probabilities(input);

        let r = self.rng.gen::<f64>();
        match ps.iter().position(|p| *p > r) {
            Some(index) => index,
            None => ps.len() - 1,
        }
    }

    fn probability(&mut self, input: &S, a: usize) -> f64 { self.probabilities(input)[a] }
}

impl<S, M: Projector<S>> FinitePolicy<S> for Gibbs<S, M> {
    fn probabilities(&mut self, input: &S) -> Vector<f64> {
        let values = self.fa.evaluate(input).unwrap();

        probabilities_from_values(values.as_slice().unwrap())
    }
}

impl<S, M: Projector<S>> DifferentiablePolicy<S> for Gibbs<S, M> {
    fn grad_log(&self, input: &S, a: usize) -> Matrix<f64> {
        let phi = self.fa.projector.project(input);

        let values = self.fa.approximator.evaluate(&phi).unwrap();
        let n_actions = values.len();
        let probabilities = probabilities_from_values(values.as_slice().unwrap())
            .into_shape((1, n_actions))
            .unwrap();

        let dim = self.fa.projector.dim();
        let phi = phi.expanded(self.fa.projector.dim());

        let mut grad_log = phi
            .clone()
            .into_shape((dim, 1))
            .unwrap()
            .dot(&-probabilities);
        grad_log.column_mut(a).add_assign(&phi);

        grad_log
    }
}

impl<S, M: Projector<S>> Parameterised for Gibbs<S, M> {
    fn weights(&self) -> Matrix<f64> { self.fa.approximator.weights.clone() }
}

impl<S, M: Projector<S>> ParameterisedPolicy<S> for Gibbs<S, M> {
    fn update(&mut self, input: &S, a: usize, error: f64) {
        let pi = self.probability(input, a);
        let grad_log = self.grad_log(input, a);

        self.fa
            .approximator
            .weights
            .scaled_add(pi * error, &grad_log);
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.fa.approximator.weights.add_assign(&errors)
    }
}
