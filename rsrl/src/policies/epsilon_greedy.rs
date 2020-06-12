use crate::{
    policies::{Greedy, Policy, Random},
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
pub struct EpsilonGreedy<Q> {
    #[weights]
    greedy: Greedy<Q>,
    random: Random,

    pub epsilon: f64,
}

impl<Q> EpsilonGreedy<Q> {
    pub fn new(greedy: Greedy<Q>, random: Random, epsilon: f64) -> Self {
        EpsilonGreedy {
            greedy,
            random,

            epsilon,
        }
    }
}

impl<S, Q> Function<(S,)> for EpsilonGreedy<Q>
where Q: Enumerable<(S,), Output = Vec<f64>>
{
    type Output = Vec<f64>;

    fn evaluate(&self, (s,): (S,)) -> Vec<f64> {
        let prs = self.greedy.evaluate((s,));
        let pr = self.epsilon / prs.len() as f64;

        prs.into_iter()
            .map(|p| pr + p * (1.0 - self.epsilon))
            .collect()
    }
}

impl<S, A, Q> Function<(S, A)> for EpsilonGreedy<Q>
where
    A: std::borrow::Borrow<usize>,
    Q: Enumerable<(S,), Output = Vec<f64>>,
{
    type Output = f64;

    fn evaluate(&self, (s, a): (S, A)) -> f64 {
        let prs = self.greedy.evaluate((s,));
        let pr = self.epsilon / prs.len() as f64;

        pr + (1.0 - self.epsilon) * prs[*a.borrow()]
    }
}

impl<S, Q> Enumerable<(S,)> for EpsilonGreedy<Q>
where Q: Enumerable<(S,), Output = Vec<f64>>
{
    fn evaluate_index(&self, (s,): (S,), index: usize) -> f64 { self.evaluate((s, index)) }
}

impl<S, Q> Policy<S> for EpsilonGreedy<Q>
where Q: Enumerable<(S,), Output = Vec<f64>>
{
    type Action = usize;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, s: S) -> usize {
        if rng.gen_bool(self.epsilon) {
            self.random.sample(rng, s)
        } else {
            self.greedy.sample(rng, s)
        }
    }

    fn mode(&self, s: S) -> usize { self.greedy.mode(s) }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use crate::{
        fa::mocking::MockQ,
        policies::{EnumerablePolicy, EpsilonGreedy, Policy, Greedy, Random},
        Function,
    };
    use rand::thread_rng;

    #[test]
    fn test_sampling() {
        let mut rng = thread_rng();

        let q = MockQ::new_shared(Some(vec![1.0, 0.0].into()));
        let p = EpsilonGreedy::new(Greedy::new(q), Random::new(2), 0.5);

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
        let q = MockQ::new_shared(None);
        let p = EpsilonGreedy::new(Greedy::new(q), Random::new(5), 0.5);

        p.evaluate((vec![1.0, 0.0, 0.0, 0.0, 0.0],))
            .into_iter()
            .zip([0.6, 0.1, 0.1, 0.1, 0.1].iter())
            .for_each(|(x, y)| assert_abs_diff_eq!(x, y, epsilon = 1e-6));

        p.evaluate((vec![0.0, 0.0, 0.0, 0.0, 1.0],))
            .into_iter()
            .zip([0.1, 0.1, 0.1, 0.1, 0.6].iter())
            .for_each(|(x, y)| assert_abs_diff_eq!(x, y, epsilon = 1e-6));

        p.evaluate((vec![1.0, 0.0, 0.0, 0.0, 1.0],))
            .into_iter()
            .zip([0.35, 0.1, 0.1, 0.1, 0.35].iter())
            .for_each(|(x, y)| assert_abs_diff_eq!(x, y, epsilon = 1e-6));
    }

    #[test]
    fn test_probabilites_uniform() {
        let q = MockQ::new_shared(Some(vec![-1.0, 0.0, 0.0, 0.0].into()));
        let p = EpsilonGreedy::new(Greedy::new(q), Random::new(4), 1.0);

        p.evaluate((vec![-1.0, 0.0, 0.0, 0.0],))
            .into_iter()
            .zip([0.25, 0.25, 0.25, 0.25].iter())
            .for_each(|(x, y)| assert_abs_diff_eq!(x, y, epsilon = 1e-6));
    }
}
