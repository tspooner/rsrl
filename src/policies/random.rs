use super::Policy;
use geometry::{Space, ActionSpace};
use rand::{thread_rng, ThreadRng};
use rand::distributions::{Range, IndependentSample};


pub struct Random {
    rng: ThreadRng,
}

impl Random {
    pub fn new() -> Self {
        Random { rng: thread_rng() }
    }
}

impl Policy for Random {
    fn sample(&mut self, qs: &[f64]) -> usize {
        Range::new(0, qs.len()).ind_sample(&mut self.rng)
    }

    fn probabilities(&mut self, qs: &[f64]) -> Vec<f64> {
        vec![1.0/qs.len() as f64; qs.len()]
    }
}
