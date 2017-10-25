

use super::Policy;
use geometry::{Space, ActionSpace};
use rand::{thread_rng, ThreadRng};


pub struct Random {
    action_space: ActionSpace,
    rng: ThreadRng,
}

impl Random {
    pub fn new(aspace: ActionSpace) -> Self {
        Random {
            action_space: aspace,
            rng: thread_rng(),
        }
    }
}

impl Policy for Random {
    fn sample(&mut self, _: &[f64]) -> usize {
        self.action_space.sample(&mut self.rng)
    }

    fn probabilities(&mut self, qs: &[f64]) -> Vec<f64> {
        vec![1.0/qs.len() as f64; qs.len()]
    }
}
