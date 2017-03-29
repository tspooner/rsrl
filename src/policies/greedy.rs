use super::Policy;


pub struct Greedy;

// TODO: Implement random selection of matching maxima.
impl Policy for Greedy {
    fn sample(&mut self, qs: &[f64]) -> usize {
        qs.iter().enumerate().skip(1)
           .fold((0, qs[0]), |s, x| if *x.1 > s.1 {(x.0, x.1.clone())} else {s}).0
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

        let v = vec![0.00000000001, 0.00000000002];
        assert!(g.sample(&v) == 1);
    }
}
