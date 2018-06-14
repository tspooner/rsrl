use core::{Handler};
use domains::Transition;
use fa::SharedQFunction;
use geometry::Vector;
use policies::{Policy, FinitePolicy};
use rand::{Rng, thread_rng};
use utils::argmaxima;

pub struct Greedy<S>(SharedQFunction<S>);

impl<S> Greedy<S> {
    pub fn new(q_func: SharedQFunction<S>) -> Self {
        Greedy(q_func)
    }
}

impl<S> Handler<Transition<S, usize>> for Greedy<S> {}

impl<S> Policy<S, usize> for Greedy<S> {
    fn sample(&mut self, s: &S) -> usize {
        let maxima = argmaxima(self.0.borrow().evaluate(s).unwrap().as_slice().unwrap()).1;

        if maxima.len() == 1 {
            maxima[0]
        } else {
            *thread_rng()
                .choose(&maxima)
                .expect("No valid actions to choose from in `Greedy.sample_qs(qs)`")
        }
    }

    fn probability(&mut self, s: &S, a: usize) -> f64 {
        self.probabilities(s)[a]
    }
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
    use super::{Greedy, Policy};

    #[test]
    #[should_panic]
    fn test_0d() { Greedy.sample(&vec![]); }

    #[test]
    fn test_1d() {
        let mut g = Greedy;

        let mut v = vec![1.0];
        assert!(g.sample(&v) == 0);

        v = vec![-100.0];
        assert!(g.sample(&v) == 0);
    }

    #[test]
    fn test_two_positive() {
        let mut g = Greedy;

        let mut v = vec![10.0, 1.0];
        assert!(g.sample(&v) == 0);

        v = vec![1.0, 10.0];
        assert!(g.sample(&v) == 1);
    }

    #[test]
    fn test_two_negative() {
        let mut g = Greedy;

        let mut v = vec![-10.0, -1.0];
        assert!(g.sample(&v) == 1);

        v = vec![-1.0, -10.0];
        assert!(g.sample(&v) == 0);
    }

    #[test]
    fn test_two_alt() {
        let mut g = Greedy;

        let mut v = vec![10.0, -1.0];
        assert!(g.sample(&v) == 0);

        v = vec![-10.0, 1.0];
        assert!(g.sample(&v) == 1);

        v = vec![1.0, -10.0];
        assert!(g.sample(&v) == 0);

        v = vec![-1.0, 10.0];
        assert!(g.sample(&v) == 1);
    }

    #[test]
    fn test_long() {
        let mut g = Greedy;

        let v = vec![-123.1, 123.1, 250.5, -1240.0, -4500.0, 10000.0, 20.1];
        assert!(g.sample(&v) == 5);
    }

    #[test]
    fn test_precision() {
        let mut g = Greedy;

        let v = vec![1e-7, 2e-7];
        assert!(g.sample(&v) == 1);
    }

    #[test]
    fn test_probabilites() {
        let mut g = Greedy;

        assert_eq!(
            g.probabilities(&[1e-7, 2e-7, 3e-7, 4e-7]),
            vec![0.0, 0.0, 0.0, 1.0]
        );

        assert_eq!(
            g.probabilities(&[1e-7, 1e-7, 1e-7, 1e-7]),
            vec![0.25, 0.25, 0.25, 0.25]
        );
    }
}
