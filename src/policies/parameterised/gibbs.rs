use crate::{
    core::*,
    fa::{Approximator, Parameterised, Projector, QFunction},
    geometry::{MatrixView, MatrixViewMut},
    policies::{
        sample_probs_with_rng,
        DifferentiablePolicy,
        ParameterisedPolicy,
        FinitePolicy,
        Policy
    },
    utils::argmax_choose,
};
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

impl<S, F: QFunction<S>> Policy<S> for Gibbs<F> {
    type Action = usize;

    fn sample(&mut self, s: &S) -> usize {
        let ps = self.probabilities(s);

        sample_probs_with_rng(&mut self.rng, ps.as_slice().unwrap())
    }

    fn mpa(&mut self, s: &S) -> usize {
        let ps = self.probabilities(s);

        argmax_choose(&mut self.rng, ps.as_slice().unwrap()).1
    }

    fn probability(&mut self, input: &S, a: usize) -> f64 { self.probabilities(input)[a] }
}

impl<S, F: QFunction<S>> FinitePolicy<S> for Gibbs<F> {
    fn n_actions(&self) -> usize {
        self.fa.n_outputs()
    }

    fn probabilities(&mut self, input: &S) -> Vector<f64> {
        let values = self.fa.evaluate(&self.fa.to_features(input)).unwrap();

        probabilities_from_values(values.as_slice().unwrap())
    }
}

impl<S, F: QFunction<S>> DifferentiablePolicy<S> for Gibbs<F> {
    fn grad_log(&self, input: &S, a: usize) -> Matrix<f64> {
        let phi = self.fa.to_features(input);

        let values = self.fa.evaluate(&phi).unwrap();
        let n_actions = values.len();
        let probabilities = probabilities_from_values(values.as_slice().unwrap())
            .into_shape((1, n_actions))
            .unwrap();

        let dim = self.fa.n_features();
        let phi = phi.expanded(dim);

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
    fn weights(&self) -> Matrix<f64> {
        self.fa.weights()
    }

    fn weights_view(&self) -> MatrixView<f64> {
        self.fa.weights_view()
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        self.fa.weights_view_mut()
    }
}

impl<S, F: QFunction<S> + Parameterised> ParameterisedPolicy<S> for Gibbs<F> {
    fn update(&mut self, input: &S, a: usize, error: f64) {
        let grad_log = self.grad_log(input, a);

        self.weights_view_mut().scaled_add(error, &grad_log);
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.weights_view_mut().add_assign(&errors);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        fa::{LFA, Parameterised, basis::fixed::Polynomial},
        policies::{Policy, ParameterisedPolicy, FinitePolicy},
    };
    use super::Gibbs;

    #[test]
    fn test_probabilities() {
        let fa = LFA::vector(Polynomial::new(1, vec![(0.0, 1.0)]), 3);
        let mut p = Gibbs::new(fa);

        p.update(&vec![0.0], 0, -1.0);
        p.update(&vec![0.0], 1, 1.0);
        p.update(&vec![0.0], 2, -1.0);

        let ps = p.probabilities(&vec![0.0]);

        assert!(ps[0] < ps[1]);
        assert!(ps[2] < ps[1]);
    }
}
