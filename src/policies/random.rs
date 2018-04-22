use rand::{thread_rng, ThreadRng};
use rand::distributions::{IndependentSample, Range};
use super::{Policy, FinitePolicy};

pub struct Random {
    rng: ThreadRng,
}

impl Random {
    pub fn new() -> Self { Random { rng: thread_rng() } }
}

impl Policy<[f64], usize> for Random {
    fn sample(&mut self, q_values: &[f64]) -> usize {
        Range::new(0, q_values.len()).ind_sample(&mut self.rng)
    }

    fn probability(&mut self, q_values: &[f64], _: usize) -> f64 {
        1.0 / q_values.len() as f64
    }
}

impl FinitePolicy<[f64]> for Random {
    fn probabilities(&mut self, q_values: &[f64]) -> Vec<f64> {
        let n = q_values.len();

        vec![1.0 / n as f64; n]
    }
}

#[cfg(test)]
mod tests {
    use super::{Policy, Random};

    #[test]
    fn test_sampling() {
        let mut p = Random::new();
        let qs = vec![1.0, 0.0];

        let mut n0: f64 = 0.0;
        let mut n1: f64 = 0.0;
        for _ in 0..10000 {
            match p.sample(&qs) {
                0 => n0 += 1.0,
                _ => n1 += 1.0,
            }
        }

        assert!((0.50 - n0 / 10000.0).abs() < 0.05);
        assert!((0.50 - n1 / 10000.0).abs() < 0.05);
    }

    #[test]
    fn test_probabilites() {
        let mut p = Random::new();

        assert_eq!(p.probabilities(&[1.0, 0.0, 0.0, 1.0]), vec![0.25; 4]);
        assert_eq!(p.probabilities(&[1.0, 0.0, 0.0, 0.0, 0.0]), vec![0.2; 5]);
        assert_eq!(p.probabilities(&[0.0, 0.0, 0.0, 0.0, 1.0]), vec![0.2; 5]);
    }
}
