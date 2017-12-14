use super::Policy;

use Parameter;
use rand::{Rng, thread_rng, ThreadRng};
use std::f64;


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

impl Policy for Boltzmann {
    fn sample(&mut self, qs: &[f64]) -> usize {
        let ps = self.probabilities(qs);

        let r = self.rng.next_f64();
        match ps.iter().position(|p| *p > r) {
            Some(index) => index,
            None => ps.len() - 1,
        }
    }

    fn probabilities(&mut self, qs: &[f64]) -> Vec<f64> {
        let tau = self.tau.value();

        let mut z = 0.0;
        let ws: Vec<f64> = qs.iter().map(|q| {
            let v = (q / tau).exp();
            z += v;

            v
        }).collect();

        ws.iter().map(|w| w / z).collect()
    }

    fn handle_terminal(&mut self) {
        self.tau = self.tau.step();
    }
}


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

    fn kappa(c: f64, x: f64) -> f64 {
        c / (1.0 + (-x).exp())
    }
}

impl Policy for TruncatedBoltzmann {
    fn sample(&mut self, qs: &[f64]) -> usize {
        let ps = self.probabilities(qs);

        let r = self.rng.next_f64();
        match ps.iter().position(|p| *p > r) {
            Some(index) => index,
            None => ps.len() - 1,
        }
    }

    fn probabilities(&mut self, qs: &[f64]) -> Vec<f64> {
        let c = self.c.value();

        let mut z = 0.0;
        let ws: Vec<f64> = qs.iter().map(|q| {
            let v = TruncatedBoltzmann::kappa(c, *q).exp();
            z += v;

            v
        }).collect();

        ws.iter().map(|w| w / z).collect()
    }

    fn handle_terminal(&mut self) {
        self.c = self.c.step();
    }
}


#[cfg(test)]
mod tests {
    use super::{Policy, Boltzmann, Parameter};
    use std::f64::consts::E;
    use ndarray::arr1;

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

        assert!((counts/50000.0).all_close(&arr1(&vec![1.0/(1.0+E), E/(1.0+E)]), 1e-2));
    }

    #[test]
    fn test_probabilites() {
        assert!(arr1(&Boltzmann::new(1.0).probabilities(&vec![0.0, 1.0]))
                .all_close(&arr1(&vec![1.0/(1.0+E), E/(1.0+E)]), 1e-6));
        assert!(arr1(&Boltzmann::new(1.0).probabilities(&vec![0.0, 2.0]))
                .all_close(&arr1(&vec![1.0/(1.0+E*E), E*E/(1.0+E*E)]), 1e-6));
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
