use domains::Transition;
use geometry::Vector;
use policies::{Policy, FinitePolicy, QPolicy, sample_probs};
use rand::{thread_rng, ThreadRng};
use std::f64;
use {Handler, Parameter};

pub struct Boltzmann {
    tau: Parameter,
    rng: ThreadRng,
}

impl Boltzmann {
    pub fn new<T: Into<Parameter>>(tau: T) -> Self {
        Boltzmann {
            tau: tau.into(),
            rng: thread_rng(),
        }
    }
}

impl<S> Handler<Transition<S, usize>> for Boltzmann {
    fn handle_terminal(&mut self, _: &Transition<S, usize>) {
        self.tau = self.tau.step()
    }
}

impl<S> Policy<S, usize> for Boltzmann {
    fn sample(&mut self, s: &S) -> usize {
        let ps = self.probabilities(s);

        sample_probs(&mut self.rng, ps.as_slice().unwrap())
    }

    fn probability(&mut self, s: &S, a: usize) -> f64 {
        self.probabilities(s)[a]
    }
}

impl<S> FinitePolicy<S> for Boltzmann {
    fn probabilities(&mut self, _: &S) -> Vector<f64> {
        unimplemented!()
    }
}

impl<S> QPolicy<S> for Boltzmann {
    fn sample_qs(&mut self, s: &S, q_values: &[f64]) -> usize {
        let ps = self.probabilities_qs(s, q_values);

        sample_probs(&mut self.rng, ps.as_slice().unwrap())
    }

    fn probabilities_qs(&mut self, _: &S, q_values: &[f64]) -> Vector<f64> {
        let tau = self.tau.value();

        let mut z = 0.0;
        let ws: Vec<f64> = q_values.into_iter()
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
    use super::{Boltzmann, Parameter, Policy};
    use ndarray::arr1;
    use std::f64::consts::E;

    #[test]
    #[should_panic]
    fn test_0d() {
        let mut p = Boltzmann::new(1.0);
        p.sample(&vec![]);
    }

    #[test]
    fn test_1d() {
        let mut p = Boltzmann::new(1.0);

        for i in 1..100 {
            assert_eq!(p.sample(&vec![i as f64]), 0);
        }
    }

    #[test]
    fn test_2d() {
        let mut p = Boltzmann::new(1.0);
        let mut counts = arr1(&vec![0.0, 0.0]);

        for _ in 0..50000 {
            counts[p.sample(&vec![0.0, 1.0])] += 1.0;
        }

        assert!((counts / 50000.0).all_close(&arr1(&vec![1.0 / (1.0 + E), E / (1.0 + E)]), 1e-2));
    }

    #[test]
    fn test_probabilites() {
        assert!(
            arr1(&Boltzmann::new(1.0).probabilities(&vec![0.0, 1.0]))
                .all_close(&arr1(&vec![1.0 / (1.0 + E), E / (1.0 + E)]), 1e-6)
        );
        assert!(
            arr1(&Boltzmann::new(1.0).probabilities(&vec![0.0, 2.0])).all_close(
                &arr1(&vec![1.0 / (1.0 + E * E), E * E / (1.0 + E * E)]),
                1e-6,
            )
        );
    }

    #[test]
    fn test_terminal() {
        let mut tau = Parameter::exponential(100.0, 1.0, 0.9);
        let mut p = Boltzmann::new(tau);

        for _ in 0..100 {
            tau = tau.step();
            p.handle_terminal();

            assert_eq!(tau.value(), p.tau.value());
        }
    }
}
