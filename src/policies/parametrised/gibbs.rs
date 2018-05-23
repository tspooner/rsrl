use fa::{Projector, Approximator, MultiLFA};
use geometry::{Vector, Matrix};
use ndarray::Ix2;
use policies::{Policy, FinitePolicy, DifferentiablePolicy, ParameterisedPolicy};
use rand::{thread_rng, Rng, ThreadRng};
use std::f64;
use std::ops::AddAssign;
use {Transition, Handler};

pub struct Gibbs<S, M: Projector<S>> {
    pub fa: MultiLFA<S, M>,

    rng: ThreadRng,
}

impl<S, M: Projector<S>> Gibbs<S, M> {
    pub fn new(fa: MultiLFA<S, M>) -> Self {
        Gibbs {
            fa: fa,

            rng: thread_rng(),
        }
    }

    fn probabilities_from_values(values: &[f64]) -> Vector<f64> {
        let mut z = 0.0;
        let ws: Vec<f64> = values.iter()
            .map(|v| {
                let v = v.exp();
                z += v;

                v
            })
            .collect();

        ws.iter().map(|w| w / z).collect()
    }
}

impl<S, M: Projector<S>> Handler<Transition<S, usize>> for Gibbs<S, M> {}

impl<S, M: Projector<S>> Policy<S, usize> for Gibbs<S, M> {
    fn sample(&mut self, input: &S) -> usize {
        let ps = self.probabilities(input);

        let r = self.rng.next_f64();
        match ps.iter().position(|p| *p > r) {
            Some(index) => index,
            None => ps.len() - 1,
        }
    }

    fn probability(&mut self, input: &S, a: usize) -> f64 {
        self.probabilities(input)[a]
    }
}

impl<S, M: Projector<S>> FinitePolicy<S> for Gibbs<S, M> {
    fn probabilities(&mut self, input: &S) -> Vector<f64> {
        let values = self.fa.evaluate(input).unwrap();

        Gibbs::<S, M>::probabilities_from_values(values.as_slice().unwrap())
    }
}

impl<S, M: Projector<S>> DifferentiablePolicy<S, usize> for Gibbs<S, M> {
    fn grad_log(&self, input: &S, a: usize) -> Matrix<f64> {
        let phi = self.fa.projector.project(input);

        let values = self.fa.approximator.evaluate(&phi).unwrap();
        let probabilities = Gibbs::<S, M>::probabilities_from_values(values.as_slice().unwrap());

        let phi = phi.expanded(self.fa.projector.dim()).into_dimensionality::<Ix2>().unwrap();

        let mut grad_log = -probabilities.into_dimensionality::<Ix2>().unwrap() * &phi;
        grad_log.column_mut(a).add_assign(&phi);
        grad_log
    }
}

impl<S, M: Projector<S>> ParameterisedPolicy<S, usize> for Gibbs<S, M> {
    fn update(&mut self, input: &S, a: usize, error: f64) {
        let grad_log = self.grad_log(input, a);

        self.fa.approximator.weights.scaled_add(error, &grad_log)
    }

    fn update_raw(&mut self, errors: Matrix<f64>) {
        self.fa.approximator.weights.add_assign(&errors)
    }
}
