use core::*;
use domains::Transition;
use fa::SharedQFunction;
use policies::{sample_probs, FinitePolicy, Policy};
use rand::{rngs::ThreadRng, thread_rng};
use std::f64;

pub struct Boltzmann<S> {
    q_func: SharedQFunction<S>,

    tau: Parameter,
    rng: ThreadRng,
}

impl<S> Boltzmann<S> {
    pub fn new<T: Into<Parameter>>(q_func: SharedQFunction<S>, tau: T) -> Self {
        Boltzmann {
            q_func,

            tau: tau.into(),
            rng: thread_rng(),
        }
    }
}

impl<S> Algorithm for Boltzmann<S> {
    fn step_hyperparams(&mut self) { self.tau = self.tau.step() }
}

impl<S> Policy<S> for Boltzmann<S> {
    type Action = usize;

    fn sample(&mut self, s: &S) -> usize {
        let ps = self.probabilities(s);

        sample_probs(&mut self.rng, ps.as_slice().unwrap())
    }

    fn probability(&mut self, s: &S, a: usize) -> f64 { self.probabilities(s)[a] }
}

impl<S> FinitePolicy<S> for Boltzmann<S> {
    fn probabilities(&mut self, s: &S) -> Vector<f64> {
        let tau = self.tau.value();

        let mut z = 0.0;
        let ws: Vec<f64> = self
            .q_func
            .borrow()
            .evaluate(s)
            .unwrap()
            .into_iter()
            .map(|v| {
                let v = (v / tau).exp();
                z += v;

                v
            })
            .collect();

        ws.iter().map(|w| w / z).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Boltzmann, FinitePolicy, Parameter, Policy};
    use domains::{Domain, MountainCar};
    use fa::mocking::MockQ;
    use geometry::Vector;
    use std::f64::consts::E;

    #[test]
    #[should_panic]
    fn test_0d() {
        let mut p = Boltzmann::new(MockQ::new_shared(None), 1.0);

        p.sample(&vec![].into());
    }

    #[test]
    fn test_1d() {
        let mut p = Boltzmann::new(MockQ::new_shared(None), 1.0);

        for i in 1..100 {
            assert_eq!(p.sample(&vec![i as f64].into()), 0);
        }
    }

    #[test]
    fn test_2d() {
        let mut p = Boltzmann::new(MockQ::new_shared(None), 1.0);
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
    fn test_probabilites() {
        let mut p = Boltzmann::new(MockQ::new_shared(None), 1.0);

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
    fn test_terminal() {
        let mut domain = MountainCar::default();
        let mut tau = Parameter::exponential(100.0, 1.0, 0.9);
        let mut p = Boltzmann::new(MockQ::new_shared(None), tau);

        for _ in 0..100 {
            tau = tau.step();
            p.handle_terminal(&domain.step(0));

            assert_eq!(tau.value(), p.tau.value());
        }
    }
}
