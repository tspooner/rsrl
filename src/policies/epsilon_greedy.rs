use rand::{Rng, thread_rng, ThreadRng};

use super::{Policy, Greedy};
use super::random::Uniform;
use geometry::ActionSpace;


pub struct EpsilonGreedy {
    greedy: Greedy,
    random: Uniform,

    epsilon: f64,
    rng: ThreadRng,
}

impl EpsilonGreedy {
    pub fn new(action_space: ActionSpace, epsilon: f64) -> Self {
        EpsilonGreedy {
            greedy: Greedy,
            random: Uniform::new(action_space),

            epsilon: epsilon,
            rng: thread_rng(),
        }
    }
}

impl Policy for EpsilonGreedy {
    fn sample(&mut self, qs: &[f64]) -> usize {
        if self.rng.next_f64() < self.epsilon {
            self.random.sample(qs)
        } else {
            self.greedy.sample(qs)
        }
    }

    fn probabilities(&mut self, qs: &[f64]) -> Vec<f64> {
        let mut ps = vec![self.epsilon/qs.len() as f64; qs.len()];
        ps[self.sample(qs)] = 1.0 - self.epsilon;

        ps
    }
}
