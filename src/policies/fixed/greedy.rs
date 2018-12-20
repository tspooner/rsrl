use crate::core::*;
use crate::fa::SharedQFunction;
use crate::policies::{FinitePolicy, Policy};
use rand::{thread_rng, Rng, seq::SliceRandom};
use crate::utils::argmaxima;

pub struct Greedy<S>(SharedQFunction<S>);

impl<S> Greedy<S> {
    pub fn new(q_func: SharedQFunction<S>) -> Self { Greedy(q_func) }
}

impl<S> Algorithm for Greedy<S> {}

impl<S> Policy<S> for Greedy<S> {
    type Action = usize;

    fn sample(&mut self, s: &S) -> usize {
        let qs = self.0.borrow().evaluate(s).unwrap();
        let maxima = argmaxima(qs.as_slice().unwrap()).1;

        if maxima.len() == 1 {
            maxima[0]
        } else {
            *maxima
                .choose(&mut thread_rng())
                .expect("No valid actions to choose from in `Greedy.sample(qs)`")
        }
    }

    fn probability(&mut self, s: &S, a: usize) -> f64 { self.probabilities(s)[a] }
}

impl<S> FinitePolicy<S> for Greedy<S> {
    fn probabilities(&mut self, s: &S) -> Vector<f64> {
        let qs = self.0.borrow().evaluate(s).unwrap();
        let mut ps = vec![0.0; qs.len()];

        let maxima = argmaxima(qs.as_slice().unwrap()).1;

        let p = 1.0 / maxima.len() as f64;
        for i in maxima {
            ps[i] = p;
        }

        ps.into()
    }
}

#[cfg(test)]
mod tests {
    use super::{FinitePolicy, Greedy, Policy};
    use crate::fa::mocking::MockQ;
    use crate::geometry::Vector;

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
