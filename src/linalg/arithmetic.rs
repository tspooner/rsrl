use geometry::Matrix;
use super::{Gradient, Entry};

/// Apply negation to the output.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Negate<G>(G);

impl<G> Negate<G> {
    pub fn new(x: G) -> Self { Negate(x) }
}

impl<G: Into<Matrix<f64>>> Into<Matrix<f64>> for Negate<G> {
    fn into(self) -> Matrix<f64> { -self.0.into() }
}

impl<G: Gradient> Gradient for Negate<G> {
    fn dim(&self) -> [usize; 2] { self.0.dim() }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.0.map_inplace(|x| f(-x));
    }

    fn combine(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        unimplemented!()
    }

    fn for_each(&self, mut f: impl FnMut(Entry)) {
        self.0.for_each(|mut pd| {
            pd.gradient = -pd.gradient;

            f(pd);
        })
    }
}

/// Sum the output of two `Projector` instances.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Sum<G1, G2>(G1, G2);

impl<G1, G2> Sum<G1, G2> {
    pub fn new(lhs: G1, rhs: G2) -> Self {
        Sum(lhs, rhs)
    }
}

impl<G1: Into<Matrix<f64>>, G2: Into<Matrix<f64>>> Into<Matrix<f64>> for Sum<G1, G2> {
    fn into(self) -> Matrix<f64> { self.0.into() + self.1.into() }
}

impl<G1: Gradient, G2: Gradient> Gradient for Sum<G1, G2> {
    fn dim(&self) -> [usize; 2] { self.0.dim() }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.0.map_inplace(|x| f(-x));
    }

    fn combine(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        unimplemented!()
    }

    fn for_each(&self, mut f: impl FnMut(Entry)) {
        self.0.for_each(|mut pd| {
            pd.gradient = -pd.gradient;

            f(pd);
        })
    }
}

// /// Multiply the output of two `Projector` instances.
// #[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
// #[derive(Copy, Clone, Debug)]
// pub struct Product<P1, P2>(P1, P2);

// impl<P1, P2> Product<P1, P2> {
    // pub fn new(p1: P1, p2: P2) -> Self { Product(p1, p2) }
// }

// impl<P1: Projector, P2: Projector> Projector for Product<P1, P2>
// where
    // P1::Features: Mul<P2::Features>,
    // <P1::Features as Mul<P2::Features>>::Output: Features,
// {
    // type Features = <P1::Features as Mul<P2::Features>>::Output;

    // fn n_features(&self) -> usize { self.0.n_features().min(self.1.n_features()) }

    // fn project(&self, input: &[f64]) -> Self::Features {
        // self.0.project(input) * self.1.project(input)
    // }
// }

// #[cfg(test)]
// mod tests {
    // use crate::{
        // basis::fixed::Constant,
        // geometry::Vector,
    // };
    // use quickcheck::quickcheck;
    // use super::*;

    // #[test]
    // fn test_negate() {
        // fn prop_output(length: usize, value: f64) -> bool {
            // let p = Negate::new(Constant::new(length, value));
            // let f: Vector<_> = p.project(&[0.0]).into();

            // f.into_iter().all(|&v| v == -value)
        // }

        // quickcheck(prop_output as fn(usize, f64) -> bool);
    // }

    // #[test]
    // fn test_addition() {
        // fn prop_output(length: usize, v1: f64, v2: f64) -> bool {
            // let p = Sum::new(Constant::new(length, v1), Constant::new(length, v2));
            // let f: Vector<_> = p.project(&[0.0]).into();

            // f.into_iter().all(|&v| v == v1 + v2)
        // }

        // quickcheck(prop_output as fn(usize, f64, f64) -> bool);
    // }

    // #[test]
    // fn test_subtraction() {
        // fn prop_output(length: usize, v1: f64, v2: f64) -> bool {
            // let p = Sum::new(Constant::new(length, v1), Negate::new(Constant::new(length, v2)));
            // let f: Vector<_> = p.project(&[0.0]).into();

            // f.into_iter().all(|&v| v == v1 - v2)
        // }

        // quickcheck(prop_output as fn(usize, f64, f64) -> bool);
    // }

    // #[test]
    // fn test_multiplication() {
        // fn prop_output(length: usize, v1: f64, v2: f64) -> bool {
            // let p = Product::new(Constant::new(length, v1), Constant::new(length, v2));
            // let f: Vector<_> = p.project(&[0.0]).into();

            // f.into_iter().all(|&v| (v - v1 * v2) < 1e-7)
        // }

        // quickcheck(prop_output as fn(usize, f64, f64) -> bool);
    // }
// }
