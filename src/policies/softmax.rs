use crate::{
    core::*,
    domains::Transition,
    fa::{Approximator, Parameterised, QFunction, Features},
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
use ndarray::Axis;
use rand::{rngs::ThreadRng, thread_rng};
use std::{f64, ops::AddAssign};

fn probabilities_from_values<'a>(values: impl Iterator<Item = &'a f64>, tau: f64) -> Vec<f64> {
    let mut z = 0.0;

    let ps: Vec<f64> = values
        .map(|v| {
            let v = (v / tau).exp();
            z += v;

            v
        })
        .collect();

    ps.into_iter().map(|v| (v / z).min(f64::MAX)).collect()
}

pub type Gibbs<F> = Softmax<F>;

pub struct Softmax<F> {
    fa: F,
    tau: Parameter,
    rng: ThreadRng,
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
            rng: thread_rng(),
        }
    }

    pub fn standard(fa: F) -> Self {
        Self::new(fa, 1.0)
    }

    fn grad_log_phi<S>(&self, phi: &Features, a: usize) -> Matrix<f64>
        where F: QFunction<S>,
    {
        // (A x 1)
        let values = self.fa.evaluate(&phi).unwrap();
        let probabilities = probabilities_from_values(values.into_iter(), self.tau.value());

        // (N x A)
        let mut jac = self.fa.jacobian(&phi);

        // (N x 1)
        let phi = phi.expanded(self.fa.n_features());

        for (mut col, prob) in jac.gencolumns_mut().into_iter().zip(probabilities.into_iter()) {
            col.scaled_add(-prob, &phi);
        }

        jac.column_mut(a).add_assign(&phi);

        jac
    }
}

impl<F> Algorithm for Softmax<F> {
    fn handle_terminal(&mut self) { self.tau = self.tau.step(); }
}

impl<S, F: QFunction<S>> Policy<S> for Softmax<F> {
    type Action = usize;

    fn sample(&mut self, s: &S) -> usize {
        let ps = self.probabilities(s);

        sample_probs_with_rng(&mut self.rng, ps.as_slice().unwrap())
    }

    fn mpa(&mut self, s: &S) -> usize {
        let ps = self.probabilities(s);

        argmax_choose(&mut self.rng, ps.as_slice().unwrap()).1
    }

    fn probability(&mut self, s: &S, a: usize) -> f64 { self.probabilities(s)[a] }
}

impl<S, F: QFunction<S>> FinitePolicy<S> for Softmax<F> {
    fn n_actions(&self) -> usize { self.fa.n_outputs() }

    fn probabilities(&mut self, s: &S) -> Vector<f64> {
        self.fa
            .evaluate(&self.fa.embed(s))
            .map(|qs| probabilities_from_values(qs.into_iter(), self.tau.value()).into())
            .unwrap()
    }
}

impl<S, F: QFunction<S>> DifferentiablePolicy<S> for Softmax<F> {
    fn grad_log(&self, input: &S, a: usize) -> Matrix<f64> {
        self.grad_log_phi(&self.fa.embed(input), a)
    }
}

impl<F: Parameterised> Parameterised for Softmax<F> {
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

impl<S, F: QFunction<S> + Parameterised> ParameterisedPolicy<S> for Softmax<F> {
    fn update(&mut self, input: &S, a: usize, error: f64) {
        let gl = self.grad_log_phi(&self.fa.embed(input), a);

        self.fa.weights_view_mut().scaled_add(error, &gl);
    }
}

#[cfg(test)]
mod tests {
    use super::{Algorithm, Softmax, FinitePolicy, ParameterisedPolicy, Parameter, Policy};
    use crate::{
        domains::{Domain, MountainCar},
        fa::{Composable, LFA, basis::fixed::Polynomial, mocking::MockQ},
        geometry::Vector,
    };
    use std::f64::consts::E;

    #[test]
    #[should_panic]
    fn test_0d() {
        let mut p = Softmax::new(MockQ::new_shared(None), 1.0);

        p.sample(&vec![].into());
    }

    #[test]
    fn test_1d() {
        let mut p = Softmax::new(MockQ::new_shared(None), 1.0);

        for i in 1..100 {
            assert_eq!(p.sample(&vec![i as f64].into()), 0);
        }
    }

    #[test]
    fn test_2d() {
        let mut p = Softmax::new(MockQ::new_shared(None), 1.0);
        let mut counts = Vector::from_vec(vec![0.0, 0.0]);

        for _ in 0..50000 {
            counts[p.sample(&vec![0.0, 1.0].into())] += 1.0;
        }

        assert!((counts / 50000.0).all_close(
            &Vector::from_vec(vec![1.0 / (1.0 + E), E / (1.0 + E)]),
            1e-2
        ));
    }

    #[test]
    fn test_probabilites_1() {
        let mut p = Softmax::new(MockQ::new_shared(None), 1.0);

        assert!(&p.probabilities(&vec![0.0, 1.0].into()).all_close(
            &Vector::from_vec(vec![1.0 / (1.0 + E), E / (1.0 + E)]),
            1e-6,
        ));
        assert!(p.probabilities(&vec![0.0, 2.0].into()).all_close(
            &Vector::from_vec(vec![1.0 / (1.0 + E * E), E * E / (1.0 + E * E)]),
            1e-6,
        ));
    }

    #[test]
    fn test_probabilities_2() {
        let fa = LFA::vector(Polynomial::new(1, vec![(0.0, 1.0)]).with_constant(), 3);
        let mut p = Softmax::standard(fa);

        p.update(&vec![0.0], 0, -1.0);
        p.update(&vec![0.0], 1, 1.0);
        p.update(&vec![0.0], 2, -1.0);

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
