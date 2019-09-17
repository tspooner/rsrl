use crate::{
    Algorithm, Parameter,
    fa::EnumerableStateActionFunction,
    policies::{FinitePolicy, Greedy, Policy, Random}
};
use rand::Rng;

pub struct EpsilonGreedy<Q> {
    greedy: Greedy<Q>,
    random: Random,

    epsilon: Parameter,
}

impl<Q> EpsilonGreedy<Q> {
    pub fn new<T: Into<Parameter>>(greedy: Greedy<Q>, random: Random, epsilon: T) -> Self {
        EpsilonGreedy {
            greedy,
            random,

            epsilon: epsilon.into(),
        }
    }

    #[allow(non_snake_case)]
    pub fn from_Q<S, T: Into<Parameter>>(q_func: Q, epsilon: T) -> Self
    where Q: EnumerableStateActionFunction<S> {
        let greedy = Greedy::new(q_func);
        let random = Random::new(greedy.n_actions());

        EpsilonGreedy::new(greedy, random, epsilon)
    }
}

impl<Q> Algorithm for EpsilonGreedy<Q> {
    fn handle_terminal(&mut self) {
        self.epsilon = self.epsilon.step();

        self.greedy.handle_terminal();
        self.random.handle_terminal();
    }
}

impl<S, Q: EnumerableStateActionFunction<S>> Policy<S> for EpsilonGreedy<Q> {
    type Action = usize;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, s: &S) -> usize {
        if rng.gen_bool(self.epsilon.value()) {
            self.random.sample(rng, s)
        } else {
            self.greedy.sample(rng, s)
        }
    }

    fn mpa(&self, s: &S) -> usize { self.greedy.mpa(s) }

    fn probability(&self, s: &S, a: &usize) -> f64 { self.probabilities(s)[*a] }
}

impl<S, Q: EnumerableStateActionFunction<S>> FinitePolicy<S> for EpsilonGreedy<Q> {
    fn n_actions(&self) -> usize { self.greedy.n_actions() }

    fn probabilities(&self, s: &S) -> Vec<f64> {
        let prs = self.greedy.probabilities(s);
        let pr = self.epsilon / prs.len() as f64;

        prs.into_iter().map(|p| pr + p * (1.0 - self.epsilon)).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        domains::{Domain, MountainCar},
        fa::mocking::MockQ,
        geometry::Vector,
        utils::compare_floats,
    };
    use rand::thread_rng;
    use super::{Algorithm, EpsilonGreedy, FinitePolicy, Parameter, Policy};

    #[test]
    fn test_sampling() {
        let q = MockQ::new_shared(Some(vec![1.0, 0.0].into()));
        let p = EpsilonGreedy::from_Q(q.clone(), 0.5);
        let mut rng = thread_rng();

        let mut n0: f64 = 0.0;
        let mut n1: f64 = 0.0;
        for _ in 0..10000 {
            match p.sample(&mut rng, &vec![].into()) {
                0 => n0 += 1.0,
                _ => n1 += 1.0,
            }
        }

        assert!((0.75 - n0 / 10000.0).abs() < 0.05);
        assert!((0.25 - n1 / 10000.0).abs() < 0.05);
    }

    #[test]
    fn test_probabilites() {
        let q = MockQ::new_shared(Some(vec![0.0, 0.0, 0.0, 0.0, 0.0].into()));
        let p = EpsilonGreedy::from_Q(q.clone(), 0.5);

        q.borrow_mut().clear_output();

        assert!(compare_floats(
            p.probabilities(&vec![1.0, 0.0, 0.0, 0.0, 0.0]),
            &[0.6, 0.1, 0.1, 0.1, 0.1],
            1e-6
        ));

        assert!(compare_floats(
            p.probabilities(&vec![0.0, 0.0, 0.0, 0.0, 1.0]),
            &[0.1, 0.1, 0.1, 0.1, 0.6],
            1e-6
        ));

        assert!(compare_floats(
            p.probabilities(&vec![1.0, 0.0, 0.0, 0.0, 1.0]),
            &[0.35, 0.1, 0.1, 0.1, 0.35],
            1e-6
        ));
    }

    #[test]
    fn test_probabilites_uniform() {
        let q = MockQ::new_shared(Some(vec![-1.0, 0.0, 0.0, 0.0].into()));
        let p = EpsilonGreedy::from_Q(q.clone(), 1.0);

        q.borrow_mut().clear_output();

        assert!(compare_floats(
            p.probabilities(&vec![-1.0, 0.0, 0.0, 0.0]),
            &[0.25, 0.25, 0.25, 0.25],
            1e-6
        ));
    }

    #[test]
    fn test_terminal() {
        let mut epsilon = Parameter::exponential(100.0, 1.0, 0.9);

        let q = MockQ::new_shared(Some(vec![].into()));
        let mut p = EpsilonGreedy::from_Q(q, epsilon);

        for _ in 0..100 {
            epsilon = epsilon.step();
            p.handle_terminal();

            assert_eq!(epsilon.value(), p.epsilon.value());
        }
    }
}
