use super::Policy;

use utils::argmaxima;

extern crate rand;
use rand::Rng;


pub struct Greedy;

impl Policy for Greedy {
    fn sample(&mut self, qs: &[f64]) -> usize {
        let maxima = argmaxima(qs).1;

        if maxima.len() == 1 {
            maxima[0]
        } else {
            *rand::thread_rng()
                .choose(&maxima)
                .expect("no valid actions to choose from in `Greedy::sample(qs)`")
        }
    }

    fn probabilities(&mut self, qs: &[f64]) -> Vec<f64> {
        let mut ps = vec![0.0; qs.len()];

        let maxima = argmaxima(qs).1;

        let p = 1.0 / maxima.len() as f64;
        for i in maxima {
            ps[i] = p;
        }

        ps
    }
}


#[cfg(test)]
mod tests {
    use super::{Policy, Greedy};

    #[test]
    #[should_panic]
    fn test_0d() {
        Greedy.sample(&vec![]);
    }

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

        assert_eq!(g.probabilities(&[1e-7, 2e-7, 3e-7, 4e-7]),
                   vec![0.0, 0.0, 0.0, 1.0]);

        assert_eq!(g.probabilities(&[1e-7, 1e-7, 1e-7, 1e-7]),
                   vec![0.25, 0.25, 0.25, 0.25]);
    }
}
