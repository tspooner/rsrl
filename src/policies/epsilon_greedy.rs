use rand::{thread_rng, Rng, ThreadRng};
use super::{Policy, FinitePolicy, Greedy, Random};
use Parameter;

pub struct EpsilonGreedy {
    greedy: Greedy,
    random: Random,

    epsilon: Parameter,
    rng: ThreadRng,
}

impl EpsilonGreedy {
    pub fn new<T: Into<Parameter>>(epsilon: T) -> Self {
        EpsilonGreedy {
            greedy: Greedy,
            random: Random::new(),

            epsilon: epsilon.into(),
            rng: thread_rng(),
        }
    }
}

impl Policy<[f64], usize> for EpsilonGreedy {
    fn sample(&mut self, q_values: &[f64]) -> usize {
        if self.rng.next_f64() < self.epsilon.value() {
            self.random.sample(q_values)
        } else {
            self.greedy.sample(q_values)
        }
    }

    fn probability(&mut self, q_values: &[f64], a: usize) -> f64 {
        self.probabilities(q_values)[a]
    }

    fn handle_terminal(&mut self) {
        self.epsilon = self.epsilon.step();

        self.greedy.handle_terminal();
        self.random.handle_terminal();
    }
}

impl FinitePolicy<[f64]> for EpsilonGreedy {
    fn probabilities(&mut self, q_values: &[f64]) -> Vec<f64> {
        let pr = self.epsilon / q_values.len() as f64;

        self.greedy
            .probabilities(q_values)
            .iter()
            .map(|p| pr + p * (1.0 - self.epsilon))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{EpsilonGreedy, Parameter, Policy};

    #[test]
    fn test_sampling() {
        let mut p = EpsilonGreedy::new(0.5);
        let qs = vec![1.0, 0.0];

        let mut n0: f64 = 0.0;
        let mut n1: f64 = 0.0;
        for _ in 0..10000 {
            match p.sample(&qs) {
                0 => n0 += 1.0,
                _ => n1 += 1.0,
            }
        }

        assert!((0.75 - n0 / 10000.0).abs() < 0.05);
        assert!((0.25 - n1 / 10000.0).abs() < 0.05);
    }

    #[test]
    fn test_probabilites() {
        let mut p = EpsilonGreedy::new(0.5);

        assert_eq!(
            p.probabilities(&[1.0, 0.0, 0.0, 0.0, 0.0]),
            vec![0.6, 0.1, 0.1, 0.1, 0.1]
        );

        assert_eq!(
            p.probabilities(&[0.0, 0.0, 0.0, 0.0, 1.0]),
            vec![0.1, 0.1, 0.1, 0.1, 0.6]
        );

        assert_eq!(
            p.probabilities(&[1.0, 0.0, 0.0, 0.0, 1.0]),
            vec![0.35, 0.1, 0.1, 0.1, 0.35]
        );

        let mut p = EpsilonGreedy::new(1.0);

        assert_eq!(
            p.probabilities(&[-1.0, 0.0, 0.0, 0.0]),
            vec![0.25, 0.25, 0.25, 0.25]
        );
    }

    #[test]
    fn test_terminal() {
        let mut epsilon = Parameter::exponential(100.0, 1.0, 0.9);
        let mut p = EpsilonGreedy::new(epsilon);

        for _ in 0..100 {
            epsilon = epsilon.step();
            p.handle_terminal();

            assert_eq!(epsilon.value(), p.epsilon.value());
        }
    }
}
