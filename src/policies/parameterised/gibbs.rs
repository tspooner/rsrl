use crate::core::*;
use crate::fa::{Approximator, VectorLFA, Parameterised, Projector};
use crate::policies::{DifferentiablePolicy, FinitePolicy, ParameterisedPolicy, Policy};
use rand::{rngs::ThreadRng, thread_rng, Rng};
use std::{f64, ops::AddAssign};

pub struct Gibbs<F> {
    pub fa: F,

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

impl<F> Gibbs<F> {
    pub fn new(fa: F) -> Self {
        Gibbs {
            fa,

            rng: thread_rng(),
        }
    }
}

impl<F> Algorithm for Gibbs<F> {}

impl<S, M: Projector<S>> Policy<S> for Gibbs<VectorLFA<M>> {
    type Action = usize;

    fn sample(&mut self, input: &S) -> usize {
        let ps = self.probabilities(input);

        let r = self.rng.gen::<f64>();
        match ps.iter().scan(0.0, |state, &p| {
            *state = *state + p;

            Some(*state)
        }).position(|p| p > r) {
            Some(index) => index,
            None => ps.len() - 1,
        }
    }

    fn probability(&mut self, input: &S, a: usize) -> f64 { self.probabilities(input)[a] }
}

impl<S, M: Projector<S>> FinitePolicy<S> for Gibbs<VectorLFA<M>> {
    fn n_actions(&self) -> usize {
        self.fa.n_outputs()
    }

    fn probabilities(&mut self, input: &S) -> Vector<f64> {
        let values = self.fa.evaluate(input).unwrap();

        probabilities_from_values(values.as_slice().unwrap())
    }
}

impl<S, M: Projector<S>> DifferentiablePolicy<S> for Gibbs<VectorLFA<M>> {
    fn grad_log(&self, input: &S, a: usize) -> Matrix<f64> {
        let phi = self.fa.projector.project(input);

        let values = self.fa.evaluate_primal(&phi).unwrap();
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

impl<F: Parameterised> Parameterised for Gibbs<F> {
    fn weights(&self) -> Matrix<f64> { self.fa.weights() }
}

impl<S, M: Projector<S>> ParameterisedPolicy<S> for Gibbs<VectorLFA<M>> {
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

#[cfg(test)]
mod tests {
    use crate::fa::{LFA, Parameterised, basis::fixed::Polynomial};
    use crate::policies::{Policy, ParameterisedPolicy};
    use super::Gibbs;

    #[test]
    fn test_sample() {
        let fa = LFA::vector_output(Polynomial::new(1, vec![(0.0, 1.0)]), 3);
        let mut p = Gibbs::new(fa);

        for _ in 0..10 {
            p.update(&vec![0.0], 0, -100.0);
            p.update(&vec![0.0], 1, 100.0);
            p.update(&vec![0.0], 2, -100.0);
        }

        assert_eq!(p.sample(&vec![0.0]), 1);
    }
}
