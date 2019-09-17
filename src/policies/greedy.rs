use crate::{
    Algorithm,
    fa::EnumerableStateActionFunction,
    policies::{FinitePolicy, Policy},
    utils::{argmaxima},
};

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Greedy<Q>(Q);

impl<Q> Greedy<Q> {
    pub fn new(q_func: Q) -> Self { Greedy(q_func) }

    pub fn argmax_qs(qs: &[f64]) -> usize { argmaxima(qs).1[0] }
}

impl<Q> Algorithm for Greedy<Q> {}

impl<S, Q: EnumerableStateActionFunction<S>> Policy<S> for Greedy<Q> {
    type Action = usize;

    fn mpa(&self, s: &S) -> usize {
        Greedy::<Q>::argmax_qs(&self.0.evaluate_all(s))
    }

    fn probability(&self, s: &S, a: &usize) -> f64 { self.probabilities(s)[*a] }
}

impl<S, Q: EnumerableStateActionFunction<S>> FinitePolicy<S> for Greedy<Q> {
    fn n_actions(&self) -> usize { self.0.n_actions() }

    fn probabilities(&self, s: &S) -> Vec<f64> {
        let qs = self.0.evaluate_all(s);
        let mut ps = vec![0.0; qs.len()];

        let (_, maxima) = argmaxima(&qs);

        let p = 1.0 / maxima.len() as f64;
        for i in maxima {
            ps[i] = p;
        }

        ps.into()
    }
}

#[cfg(test)]
mod tests {
    use crate::{fa::mocking::MockQ, utils::compare_floats};
    use rand::thread_rng;
    use super::{FinitePolicy, Greedy, Policy};

    #[test]
    fn test_1d() {
        let p = Greedy::new(MockQ::new_shared(None));
        let mut rng = thread_rng();

        assert!(p.sample(&mut rng, &vec![1.0]) == 0);
        assert!(p.sample(&mut rng, &vec![-100.0]) == 0);
    }

    #[test]
    fn test_two_positive() {
        let p = Greedy::new(MockQ::new_shared(None));
        let mut rng = thread_rng();

        assert!(p.sample(&mut rng, &vec![10.0, 1.0]) == 0);
        assert!(p.sample(&mut rng, &vec![1.0, 10.0]) == 1);
    }

    #[test]
    fn test_two_negative() {
        let p = Greedy::new(MockQ::new_shared(None));
        let mut rng = thread_rng();

        assert!(p.sample(&mut rng, &vec![-10.0, -1.0]) == 1);
        assert!(p.sample(&mut rng, &vec![-1.0, -10.0]) == 0);
    }

    #[test]
    fn test_two_alt() {
        let p = Greedy::new(MockQ::new_shared(None));
        let mut rng = thread_rng();

        assert!(p.sample(&mut rng, &vec![10.0, -1.0]) == 0);
        assert!(p.sample(&mut rng, &vec![-10.0, 1.0]) == 1);
        assert!(p.sample(&mut rng, &vec![1.0, -10.0]) == 0);
        assert!(p.sample(&mut rng, &vec![-1.0, 10.0]) == 1);
    }

    #[test]
    fn test_long() {
        let p = Greedy::new(MockQ::new_shared(None));
        let mut rng = thread_rng();

        assert!(p.sample(&mut rng, &vec![-123.1, 123.1, 250.5, -1240.0, -4500.0, 10000.0, 20.1]) == 5);
    }

    #[test]
    fn test_precision() {
        let p = Greedy::new(MockQ::new_shared(None));
        let mut rng = thread_rng();

        assert!(p.sample(&mut rng, &vec![1e-7, 2e-7].into()) == 1);
    }

    #[test]
    fn test_probabilites() {
        let p = Greedy::new(MockQ::new_shared(None));

        assert!(compare_floats(
            p.probabilities(&vec![1e-7, 2e-7, 3e-7, 4e-7]),
            &[0.0, 0.0, 0.0, 1.0],
            1e-6
        ));

        assert!(compare_floats(
            p.probabilities(&vec![1e-7, 1e-7, 1e-7, 1e-7]),
            &[0.25, 0.25, 0.25, 0.25],
            1e-6
        ));
    }
}
