use crate::{policies::Policy, Enumerable, Function};
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Random(usize);

impl Random {
    pub fn new(n_actions: usize) -> Self {
        Random(n_actions)
    }

    #[inline(always)]
    fn prob(&self) -> f64 {
        1.0 / self.0 as f64
    }
}

impl<S> Function<(S,)> for Random {
    type Output = Vec<f64>;

    fn evaluate(&self, _: (S,)) -> Vec<f64> {
        vec![self.prob(); self.0]
    }
}

impl<S, A: std::borrow::Borrow<usize>> Function<(S, A)> for Random {
    type Output = f64;

    fn evaluate(&self, _: (S, A)) -> f64 {
        self.prob()
    }
}

impl<S> Enumerable<(S,)> for Random {
    fn len(&self, _: (S,)) -> usize {
        self.0
    }

    fn evaluate_index(&self, _: (S,), _: usize) -> f64 {
        self.prob()
    }
}

impl<S> Policy<S> for Random {
    type Action = usize;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, _: S) -> usize {
        Uniform::new(0, self.0).sample(rng)
    }

    fn mode(&self, _: S) -> usize {
        panic!("Random policy has no mode.")
    }
}

#[cfg(test)]
mod tests {
    use crate::policies::{Policy, Random};
    use rand::thread_rng;

    #[test]
    fn test_sampling() {
        let p = Random::new(2);
        let mut rng = thread_rng();

        let qs = vec![1.0, 0.0];

        let mut n0: f64 = 0.0;
        let mut n1: f64 = 0.0;
        for _ in 0..10000 {
            match p.sample(&mut rng, &qs) {
                0 => n0 += 1.0,
                _ => n1 += 1.0,
            }
        }

        assert!((0.50 - n0 / 10000.0).abs() < 0.05);
        assert!((0.50 - n1 / 10000.0).abs() < 0.05);
    }

    // #[test]
    // fn test_probabilites() {
    // let p = Random::new(4);

    // assert!(compare_floats(
    // p.evaluate((&vec![1.0, 0.0, 0.0, 1.0],)),
    // &[0.25; 4],
    // 1e-6
    // ));

    // let p = Random::new(5);

    // assert!(compare_floats(
    // p.evaluate((&vec![1.0, 0.0, 0.0, 0.0, 0.0],)),
    // &[0.2; 5],
    // 1e-6
    // ));

    // assert!(compare_floats(
    // p.evaluate((&vec![0.0, 0.0, 0.0, 0.0, 1.0],)),
    // &[0.2; 5],
    // 1e-6
    // ));
    // }
}
