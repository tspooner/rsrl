use utils;


/// An interface for kernel functions.
pub trait Kernel {
    fn apply(&self, x1: &[f64], x2: &[f64]) -> f64;
}


/// Linear kernel function.
#[derive(Clone, Copy)]
pub struct Linear {
    c: f64,
}

impl Linear {
    pub fn new(c: f64) -> Self {
        Linear { c: c }
    }
}

impl Kernel for Linear {
    fn apply(&self, x1: &[f64], x2: &[f64]) -> f64 {
        utils::dot(x1, x2) + self.c
    }
}


/// Exponential kernel function.
#[derive(Clone, Copy)]
pub struct Exponential {
    gamma: f64,
}

impl Exponential {
    pub fn new(sigma: f64) -> Self {
        Exponential { gamma: -1.0 / (2.0 * sigma * sigma) }
    }
}

impl Kernel for Exponential {
    fn apply(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let norm = x1.iter().zip(x2.iter()).fold(0.0, |acc, (a, b)| {
            let d = a - b;
            acc + d * d
        });

        (self.gamma * norm).exp()
    }
}

// TODO: Add kernel arithmetic:
//  - Sum
//  - Product
//  - Scaling
//  - Embedding


#[cfg(test)]
mod tests {
    use super::{Kernel, Linear, Exponential};

    #[test]
    fn test_linear() {
        let l = Linear::new(0.0);

        assert_eq!(l.apply(&vec![1.0], &vec![2.0]), 2.0);
        assert_eq!(l.apply(&vec![1.0, 2.0], &vec![3.0, 4.0]), 11.0);
        assert_eq!(l.apply(&vec![1.0, 2.0, 3.0], &vec![4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn test_linear_shifted() {
        let l = Linear::new(-1.0);

        assert_eq!(l.apply(&vec![1.0], &vec![2.0]), 1.0);
        assert_eq!(l.apply(&vec![1.0, 2.0], &vec![3.0, 4.0]), 10.0);
        assert_eq!(l.apply(&vec![1.0, 2.0, 3.0], &vec![4.0, 5.0, 6.0]), 31.0);
    }

    #[test]
    fn test_exponential() {
        let e = Exponential::new(1.0);

        assert!((e.apply(&vec![1.0], &vec![2.0]) - 0.606530660).abs() < 1e-9);
        assert!((e.apply(&vec![1.0, 2.0], &vec![3.0, 4.0]) - 0.0183156389).abs() < 1e-9);
        assert!((e.apply(&vec![1.0, 2.0, 3.0], &vec![4.0, 5.0, 6.0]) - 1.371e-6).abs() < 1e-9);
    }
}
