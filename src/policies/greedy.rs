use crate::{
    core::*,
    fa::QFunction,
    policies::{FinitePolicy, Policy},
    utils::{argmax_choose, argmaxima},
};
use rand::thread_rng;

pub struct Greedy<Q>(Q);

impl<Q> Greedy<Q> {
    pub fn new(q_func: Q) -> Self { Greedy(q_func) }
}

impl<Q> Algorithm for Greedy<Q> {}

impl<S, Q: QFunction<S>> Policy<S> for Greedy<Q> {
    type Action = usize;

    fn mpa(&mut self, s: &S) -> usize {
        self.0
            .evaluate(&self.0.to_features(s))
            .map(|qs| argmax_choose(&mut thread_rng(), qs.as_slice().unwrap()).1)
            .unwrap()
    }

    fn probability(&mut self, s: &S, a: usize) -> f64 { self.probabilities(s)[a] }
}

impl<S, Q: QFunction<S>> FinitePolicy<S> for Greedy<Q> {
    fn n_actions(&self) -> usize { self.0.n_outputs() }

    fn probabilities(&mut self, s: &S) -> Vector<f64> {
        self.0
            .evaluate(&self.0.to_features(s))
            .map(|qs| {
                let mut ps = vec![0.0; qs.len()];

                let (_, maxima) = argmaxima(qs.as_slice().unwrap());

                let p = 1.0 / maxima.len() as f64;
                for i in maxima {
                    ps[i] = p;
                }

                ps.into()
            })
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::{FinitePolicy, Greedy, Policy};
    use crate::{fa::mocking::MockQ, geometry::Vector};

    #[test]
    #[should_panic]
    fn test_0d() {
        let mut p = Greedy::new(MockQ::new_shared(None));

        p.sample(&vec![].into());
    }

    #[test]
    fn test_1d() {
        let mut p = Greedy::new(MockQ::new_shared(None));

        assert!(p.sample(&vec![1.0].into()) == 0);
        assert!(p.sample(&vec![-100.0].into()) == 0);
    }

    #[test]
    fn test_two_positive() {
        let mut p = Greedy::new(MockQ::new_shared(None));

        assert!(p.sample(&vec![10.0, 1.0].into()) == 0);
        assert!(p.sample(&vec![1.0, 10.0].into()) == 1);
    }

    #[test]
    fn test_two_negative() {
        let mut p = Greedy::new(MockQ::new_shared(None));

        assert!(p.sample(&vec![-10.0, -1.0].into()) == 1);
        assert!(p.sample(&vec![-1.0, -10.0].into()) == 0);
    }

    #[test]
    fn test_two_alt() {
        let mut p = Greedy::new(MockQ::new_shared(None));

        assert!(p.sample(&vec![10.0, -1.0].into()) == 0);
        assert!(p.sample(&vec![-10.0, 1.0].into()) == 1);
        assert!(p.sample(&vec![1.0, -10.0].into()) == 0);
        assert!(p.sample(&vec![-1.0, 10.0].into()) == 1);
    }

    #[test]
    fn test_long() {
        let mut p = Greedy::new(MockQ::new_shared(None));

        assert!(p.sample(&vec![-123.1, 123.1, 250.5, -1240.0, -4500.0, 10000.0, 20.1].into()) == 5);
    }

    #[test]
    fn test_precision() {
        let mut p = Greedy::new(MockQ::new_shared(None));

        assert!(p.sample(&vec![1e-7, 2e-7].into()) == 1);
    }

    #[test]
    fn test_probabilites() {
        let mut p = Greedy::new(MockQ::new_shared(None));

        assert_eq!(
            p.probabilities(&vec![1e-7, 2e-7, 3e-7, 4e-7].into()),
            Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0])
        );

        assert_eq!(
            p.probabilities(&vec![1e-7, 1e-7, 1e-7, 1e-7].into()),
            Vector::from_vec(vec![0.25, 0.25, 0.25, 0.25])
        );
    }
}
