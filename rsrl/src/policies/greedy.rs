use crate::{
    policies::Policy,
    utils::{argmax_choose_rng, argmaxima},
    Enumerable,
    Function,
};
use rand::Rng;

#[derive(Clone, Debug, Parameterised)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Greedy<Q>(Q);

impl<Q> Greedy<Q> {
    pub fn new(q_func: Q) -> Self { Greedy(q_func) }
}

impl<S, Q> Function<(S,)> for Greedy<Q>
where Q: Enumerable<(S,), Output = Vec<f64>>
{
    type Output = Vec<f64>;

    fn evaluate(&self, (s,): (S,)) -> Vec<f64> {
        let qs = self.0.evaluate((s,));
        let mut ps = vec![0.0; qs.len()];

        let (maxima, _) = argmaxima(qs);

        let p = 1.0 / maxima.len() as f64;
        for i in maxima {
            ps[i] = p;
        }

        ps
    }
}

impl<S, A, Q> Function<(S, A)> for Greedy<Q>
where
    A: std::borrow::Borrow<usize>,
    Q: Enumerable<(S,), Output = Vec<f64>>,
{
    type Output = f64;

    fn evaluate(&self, (s, a): (S, A)) -> f64 {
        let qs = self.0.evaluate((s,));
        let (maxima, _) = argmaxima(qs);

        if maxima.contains(a.borrow()) {
            1.0 / maxima.len() as f64
        } else {
            0.0
        }
    }
}

impl<S, Q: Enumerable<(S,), Output = Vec<f64>>> Enumerable<(S,)> for Greedy<Q> {
    fn len(&self, args: (S,)) -> usize { self.0.len(args) }

    fn evaluate_index(&self, (s,): (S,), index: usize) -> f64 { self.evaluate((s, index)) }
}

impl<S, Q: Enumerable<(S,), Output = Vec<f64>>> Policy<S> for Greedy<Q> {
    type Action = usize;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, s: S) -> usize {
        let qs = self.0.evaluate((s,));

        argmax_choose_rng(rng, qs).0
    }

    fn mode(&self, s: S) -> usize { self.0.find_max((s,)).0 }
}

#[cfg(test)]
mod tests {
    use crate::{
        fa::mocking::MockQ,
        policies::{EnumerablePolicy, Greedy, Policy},
        utils::compare_floats,
        Function,
    };
    use rand::thread_rng;

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

        assert!(
            p.sample(
                &mut rng,
                &vec![-123.1, 123.1, 250.5, -1240.0, -4500.0, 10000.0, 20.1]
            ) == 5
        );
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
            p.evaluate((&vec![1e-7, 2e-7, 3e-7, 4e-7],)),
            &[0.0, 0.0, 0.0, 1.0],
            1e-6
        ));

        assert!(compare_floats(
            p.evaluate((&vec![1e-7, 1e-7, 1e-7, 1e-7],)),
            &[0.25, 0.25, 0.25, 0.25],
            1e-6
        ));
    }
}
