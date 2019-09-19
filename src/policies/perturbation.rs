use crate::core::*;
use crate::geometry::Space;
use crate::policies::{EnumerablePolicy, Policy};
use rand::{
    distributions::{Distribution, Normal},
    rngs::ThreadRng,
    Rng,
    thread_rng,
};
use std::ops::Add;

pub struct PerturbedPolicy<P, D> {
    pub base_policy: P,
    pub noise_dist: D,
}

impl<P, D> PerturbedPolicy<P, D> {
    pub fn new(base_policy: P, noise_dist: D) -> Self {
        PerturbedPolicy {
            base_policy,
            noise_dist,
        }
    }
}

impl<P> PerturbedPolicy<P, Normal> {
    pub fn normal(base_policy: P, std_dev: f64) -> Self {
        PerturbedPolicy::new(base_policy, Normal::new(0.0, std_dev))
    }
}

impl<P: Algorithm, D> Algorithm for PerturbedPolicy<P, D> {
    fn handle_terminal(&mut self) {
        self.base_policy.handle_terminal();
    }
}

impl<S, P, D> Policy<S> for PerturbedPolicy<P, D>
where
    P: Policy<S>,
    D: Distribution<P::Action>,
    P::Action: Add<P::Action, Output = P::Action>,
{
    type Action = P::Action;

    fn sample(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        let base_action = self.base_policy.sample(rng, s);
        let perturbation = self.noise_dist.sample(rng);

        base_action + perturbation
    }

    fn probability(&self, _: &S, _: &P::Action) -> f64 {
        unimplemented!()
    }
}

impl<S, P, D> EnumerablePolicy<S> for PerturbedPolicy<P, D>
where
    P: EnumerablePolicy<S>,
    D: Distribution<P::Action>,
{
    fn n_actions(&self) -> usize {
        self.base_policy.n_actions()
    }

    fn probabilities(&self, _: &S) -> Vector<f64> {
        unimplemented!()
    }
}
