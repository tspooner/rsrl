use rand::{Rng, thread_rng, ThreadRng};

use {Parameter};
use super::{Policy, Greedy, Random};
use geometry::ActionSpace;


pub struct EpsilonGreedy {
    greedy: Greedy,
    random: Random,

    epsilon: Parameter,
    rng: ThreadRng,
}

impl EpsilonGreedy {
    pub fn new<T: Into<Parameter>>(action_space: ActionSpace, epsilon: T) -> Self {
        EpsilonGreedy {
            greedy: Greedy,
            random: Random::new(action_space),

            epsilon: epsilon.into(),
            rng: thread_rng(),
        }
    }
}

impl Policy for EpsilonGreedy {
    fn sample(&mut self, qs: &[f64]) -> usize {
        if self.rng.next_f64() < self.epsilon.value() {
            self.random.sample(qs)
        } else {
            self.greedy.sample(qs)
        }
    }

    fn probabilities(&mut self, qs: &[f64]) -> Vec<f64> {
        let pr = self.epsilon / qs.len() as f64;

        self.greedy.probabilities(qs).iter().map(|p| {
            pr + p*(1.0 - self.epsilon)
        }).collect()
    }

    fn handle_terminal(&mut self) {
        self.epsilon = self.epsilon.step();

        self.greedy.handle_terminal();
        self.random.handle_terminal();
    }
}


#[cfg(test)]
mod tests {
    use policies::{Policy, EpsilonGreedy};
    use geometry::ActionSpace;
    use geometry::dimensions::Discrete;

    fn action_space(n_actions: usize) -> ActionSpace {
        ActionSpace::new(Discrete::new(n_actions))
    }

    #[test]
    #[ignore]
    fn test_sampling() {
        let mut p = EpsilonGreedy::new(action_space(2), 0.5);
        let qs = vec![1.0, 0.0];

        let mut n0: f64 = 0.0;
        let mut n1: f64 = 0.0;
        for _ in 0..50000 {
            match p.sample(&qs) {
                0 => n0 += 1.0,
                _ => n1 += 1.0
            }
        }

        assert!((0.75 - n0 / 50000.0).abs() < 0.01);
        assert!((0.25 - n1 / 50000.0).abs() < 0.01);
    }

    #[test]
    fn test_probabilites() {
        let mut p = EpsilonGreedy::new(action_space(5), 0.5);

        assert_eq!(p.probabilities(&[1.0, 0.0, 0.0, 0.0, 0.0]),
                   vec![0.6, 0.1, 0.1, 0.1, 0.1]);

        assert_eq!(p.probabilities(&[0.0, 0.0, 0.0, 0.0, 1.0]),
                   vec![0.1, 0.1, 0.1, 0.1, 0.6]);

        assert_eq!(p.probabilities(&[1.0, 0.0, 0.0, 0.0, 1.0]),
                   vec![0.35, 0.1, 0.1, 0.1, 0.35]);

        let mut p = EpsilonGreedy::new(action_space(4), 1.0);

        assert_eq!(p.probabilities(&[-1.0, 0.0, 0.0, 0.0]),
                   vec![0.25, 0.25, 0.25, 0.25]);
    }
}
