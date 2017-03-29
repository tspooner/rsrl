use rand::{Rng, thread_rng, ThreadRng};

use super::{Policy, Greedy};
use geometry::{Space, ActionSpace};


pub struct EpsilonGreedy {
    greedy: Greedy,
    action_space: ActionSpace,
    epsilon: f64,
    rng: ThreadRng,
}

impl EpsilonGreedy {
    pub fn new(action_space: ActionSpace, epsilon: f64) -> Self {
        EpsilonGreedy {
            greedy: Greedy,
            action_space: action_space,
            epsilon: epsilon,
            rng: thread_rng(),
        }
    }
}

impl Policy for EpsilonGreedy {
    fn sample(&mut self, qs: &[f64]) -> usize {
        if self.rng.next_f64() < self.epsilon {
            self.action_space.sample(&mut self.rng)
        } else {
            self.greedy.sample(qs)
        }
    }
}
