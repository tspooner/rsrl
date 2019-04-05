use super::{FinitePolicy, Greedy, Policy, Random};
use crate::{core::*, domains::Transition, fa::QFunction};
use rand::{rngs::ThreadRng, thread_rng, Rng};

pub struct EpsilonGreedy<Q> {
    greedy: Greedy<Q>,
    random: Random,

    epsilon: Parameter,
    rng: ThreadRng,
}

impl<Q> EpsilonGreedy<Q> {
    pub fn new<T: Into<Parameter>>(greedy: Greedy<Q>, random: Random, epsilon: T) -> Self {
        EpsilonGreedy {
            greedy,
            random,

            epsilon: epsilon.into(),
            rng: thread_rng(),
        }
    }

    #[allow(non_snake_case)]
    pub fn from_Q<S, T: Into<Parameter>>(q_func: Shared<Q>, epsilon: T) -> Self
    where Q: QFunction<S> {
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

impl<S, Q: QFunction<S>> Policy<S> for EpsilonGreedy<Q> {
    type Action = usize;

    fn sample(&mut self, s: &S) -> usize {
        if self.rng.gen_bool(self.epsilon.value()) {
            self.random.sample(s)
        } else {
            self.greedy.sample(s)
        }
    }

    fn mpa(&mut self, s: &S) -> usize { self.greedy.mpa(s) }

    fn probability(&mut self, s: &S, a: usize) -> f64 { self.probabilities(s)[a] }
}

impl<S, Q: QFunction<S>> FinitePolicy<S> for EpsilonGreedy<Q> {
    fn n_actions(&self) -> usize { self.greedy.n_actions() }

    fn probabilities(&mut self, s: &S) -> Vector<f64> {
        let prs = self.greedy.probabilities(s);
        let pr = self.epsilon / prs.len() as f64;

        prs.iter().map(|p| pr + p * (1.0 - self.epsilon)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Algorithm, EpsilonGreedy, FinitePolicy, Parameter, Policy};
    use crate::{
        domains::{Domain, MountainCar},
        fa::mocking::MockQ,
        geometry::Vector,
    };

    #[test]
    fn test_sampling() {
        let q = MockQ::new_shared(Some(vec![0.0, 1.0].into()));
        let mut p = EpsilonGreedy::from_Q(q.clone(), 0.5);

        q.borrow_mut().clear_output();

        let qs = vec![1.0, 0.0].into();

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
        let mut p = EpsilonGreedy::from_Q(MockQ::new_shared(None), 0.5);

        assert!(p
            .probabilities(&vec![1.0, 0.0, 0.0, 0.0, 0.0].into())
            .all_close(&vec![0.6, 0.1, 0.1, 0.1, 0.1].into(), 1e-6));

        assert!(p
            .probabilities(&vec![0.0, 0.0, 0.0, 0.0, 1.0].into())
            .all_close(&vec![0.1, 0.1, 0.1, 0.1, 0.6].into(), 1e-6));

        assert!(p
            .probabilities(&vec![1.0, 0.0, 0.0, 0.0, 1.0].into())
            .all_close(&vec![0.35, 0.1, 0.1, 0.1, 0.35].into(), 1e-6));

        let mut p = EpsilonGreedy::from_Q(MockQ::new_shared(None), 1.0);

        assert!(p
            .probabilities(&vec![-1.0, 0.0, 0.0, 0.0].into())
            .all_close(&vec![0.25, 0.25, 0.25, 0.25].into(), 1e-6));
    }

    #[test]
    fn test_terminal() {
        let mut epsilon = Parameter::exponential(100.0, 1.0, 0.9);
        let mut p = EpsilonGreedy::from_Q(MockQ::new_shared(None), epsilon);

        for _ in 0..100 {
            epsilon = epsilon.step();
            p.handle_terminal();

            assert_eq!(epsilon.value(), p.epsilon.value());
        }
    }
}
