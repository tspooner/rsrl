use domains::Transition;
use geometry::Vector;
use rand::{thread_rng, Rng, ThreadRng};
use super::{Policy, FinitePolicy, QPolicy, Greedy, Random};
use {Handler, Parameter};

pub struct EpsilonGreedy {
    random: Random,
    greedy: Greedy,

    epsilon: Parameter,
    rng: ThreadRng,
}

impl EpsilonGreedy {
    pub fn new<T: Into<Parameter>>(epsilon: T) -> Self {
        EpsilonGreedy {
            random: Random::new(),
            greedy: Greedy,

            epsilon: epsilon.into(),
            rng: thread_rng(),
        }
    }
}

impl<S> Handler<Transition<S, usize>> for EpsilonGreedy {
    fn handle_terminal(&mut self, t: &Transition<S, usize>) {
        self.epsilon = self.epsilon.step();

        self.greedy.handle_terminal(t);
        self.random.handle_terminal(t);
    }
}

impl<S> Policy<S, usize> for EpsilonGreedy {
    fn sample(&mut self, s: &S) -> usize {
        if self.rng.next_f64() < self.epsilon.value() {
            self.random.sample(s)
        } else {
            self.greedy.sample(s)
        }
    }

    fn probability(&mut self, s: &S, a: usize) -> f64 {
        self.probabilities(s)[a]
    }
}

impl<S> FinitePolicy<S> for EpsilonGreedy {
    fn probabilities(&mut self, s: &S) -> Vector<f64> {
        let prs = self.greedy.probabilities(s);
        let pr = self.epsilon / prs.len() as f64;

        prs.iter()
            .map(|p| pr + p * (1.0 - self.epsilon))
            .collect()
    }
}

impl<S> QPolicy<S> for EpsilonGreedy {
    fn sample_qs(&mut self, s: &S, q_values: &[f64]) -> usize {
        if self.rng.next_f64() < self.epsilon.value() {
            self.random.sample_qs(s, q_values)
        } else {
            self.greedy.sample_qs(s, q_values)
        }
    }

    fn probabilities_qs(&mut self, s: &S, q_values: &[f64]) -> Vector<f64> {
        let prs = self.greedy.probabilities_qs(s, q_values);
        let pr = self.epsilon / prs.len() as f64;

        prs.iter()
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
