use rand::{thread_rng, ThreadRng};

use super::Policy;
use geometry::{Space, ActionSpace};


pub struct Uniform {
    action_space: ActionSpace,
    rng: ThreadRng,
}

impl Uniform {
    pub fn new(aspace: ActionSpace) -> Self {
        Uniform {
            action_space: aspace,
            rng: thread_rng(),
        }
    }
}

impl Policy for Uniform {
    fn sample(&mut self, _: &[f64]) -> usize {
        self.action_space.sample(&mut self.rng)
    }
}
