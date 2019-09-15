use crate::{
    core::*,
    domains::Transition,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        DifferentiableStateActionFunction, EnumerableStateActionFunction,
    },
    geometry::{MatrixView, MatrixViewMut},
    policies::{
        sample_probs_with_rng,
        DifferentiablePolicy,
        FinitePolicy,
        Policy
    },
    utils::argmaxima,
};
use ndarray::{Axis, Array2, ArrayView2, ArrayViewMut2};
use rand::Rng;
use std::{f64, iter::FromIterator, ops::MulAssign};

fn softmax<C: FromIterator<f64>>(values: &[f64], tau: f64, c: f64) -> C {
    let mut z = 0.0;

    let ps: Vec<f64> = values
        .into_iter()
        .map(|v| {
            let v = ((v - c) / tau).exp();
            z += v;

            v
        })
        .collect();

    ps.into_iter().map(|v| (v / z).min(f64::MAX)).collect()
}

fn softmax_stable<C: FromIterator<f64>>(values: &[f64], tau: f64) -> C {
    let max_v = values.into_iter().fold(f64::NAN, |acc, &v| f64::max(acc, v));

    softmax(values, tau, max_v)
}

pub type Gibbs<F> = Softmax<F>;

#[derive(Parameterised)]
pub struct Softmax<F> {
    #[weights] fa: F,
    tau: Parameter,
}

impl<F> Softmax<F> {
    pub fn new<T: Into<Parameter>>(fa: F, tau: T) -> Self {
        let tau: Parameter = tau.into();

        if tau.value().abs() < 1e-7 {
            panic!("Tau parameter in Softmax must be non-zero.");
        }

        Softmax {
            fa,
            tau: tau.into(),
        }
    }

    pub fn standard(fa: F) -> Self {
        Self::new(fa, 1.0)
    }

    fn gl_matrix<S>(&self, state: &S, a: &usize) -> Matrix<f64>
        where F: EnumerableStateActionFunction<S> + DifferentiableStateActionFunction<S, usize>,
    {
        // (A x 1)
        let mut scale_factors = self.probabilities(state);
        scale_factors[*a] = scale_factors[*a] - 1.0;

        // (N x A)
        let mut jac = Matrix::zeros(self.weights_dim());

        for (ref col, sf) in (0..self.n_actions()).zip(scale_factors.into_iter()) {
            jac.scaled_add(-sf, &self.fa.grad(state, col).into());
        }

        jac
    }
}

impl<F> Algorithm for Softmax<F> {
    fn handle_terminal(&mut self) { self.tau = self.tau.step(); }
}

impl<S, F: EnumerableStateActionFunction<S>> Policy<S> for Softmax<F> {
    type Action = usize;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, s: &S) -> usize {
        sample_probs_with_rng(rng, &self.probabilities(s))
    }

    fn mpa(&self, s: &S) -> usize {
        argmaxima(&self.probabilities(s)).1[0]
    }

    fn probability(&self, s: &S, a: &usize) -> f64 { self.probabilities(s)[*a] }
}

impl<S, F: EnumerableStateActionFunction<S>> FinitePolicy<S> for Softmax<F> {
    fn n_actions(&self) -> usize { self.fa.n_actions() }

    fn probabilities(&self, s: &S) -> Vec<f64> {
        let values = self.fa.evaluate_all(s);

        softmax_stable(&values, self.tau.value())
    }
}

impl<S, F> DifferentiablePolicy<S> for Softmax<F>
where
    F: EnumerableStateActionFunction<S> + DifferentiableStateActionFunction<S, usize> + Parameterised
{
    fn update(&mut self, input: &S, a: &usize, error: f64) {
        self.fa.update_grad_scaled(&self.gl_matrix(input, a), error);
    }

    fn grad_log(&self, input: &S, a: &usize) -> Matrix<f64> { self.gl_matrix(input, a) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fa::{
            linear::{LFA, basis::{Projector, Polynomial}, optim::SGD},
            mocking::MockQ
        },
        utils::compare_floats,
    };
    use rand::thread_rng;
    use std::f64::consts::E;

    #[test]
    #[should_panic]
    fn test_0d() {
        let p = Softmax::new(MockQ::new_shared(None), 1.0);

        p.sample(&mut thread_rng(), &vec![]);
    }

    #[test]
    fn test_1d() {
        let p = Softmax::new(MockQ::new_shared(None), 1.0);
        let mut rng = thread_rng();

        for i in 1..100 {
            assert_eq!(p.sample(&mut rng, &vec![i as f64]), 0);
        }
    }

    #[test]
    fn test_2d() {
        let p = Softmax::new(MockQ::new_shared(None), 1.0);
        let mut rng = thread_rng();
        let mut counts = vec![0.0, 0.0];

        for _ in 0..50000 {
            counts[p.sample(&mut rng, &vec![0.0, 1.0])] += 1.0;
        }

        let means: Vec<f64> = counts.into_iter().map(|v| v / 50000.0).collect();

        assert!(compare_floats(means, &[1.0 / (1.0 + E), E / (1.0 + E)], 1e-2));
    }

    #[test]
    fn test_probabilites_1() {
        let p = Softmax::new(MockQ::new_shared(None), 1.0);

        assert!(compare_floats(
            p.probabilities(&vec![0.0, 1.0]),
            &[1.0 / (1.0 + E), E / (1.0 + E)],
            1e-6
        ));
        assert!(compare_floats(
            p.probabilities(&vec![0.0, 2.0]),
            &[1.0 / (1.0 + E * E), E * E / (1.0 + E * E)],
            1e-6
        ));
    }

    #[test]
    fn test_probabilities_2() {
        let fa = LFA::vector(Polynomial::new(1, 1).with_constant(), SGD(1.0), 3);
        let mut p = Softmax::standard(fa);

        p.update(&vec![0.0], &0, -5.0);
        p.update(&vec![0.0], &1, 1.0);
        p.update(&vec![0.0], &2, -5.0);

        let ps = p.probabilities(&vec![0.0]);

        assert!(ps[0] < ps[1]);
        assert!(ps[2] < ps[1]);
    }

    #[test]
    fn test_terminal() {
        let mut tau = Parameter::exponential(100.0, 1.0, 0.9);
        let mut p = Softmax::new(MockQ::new_shared(None), tau);

        for _ in 0..100 {
            tau = tau.step();
            p.handle_terminal();

            assert_eq!(tau.value(), p.tau.value());
        }
    }
}
