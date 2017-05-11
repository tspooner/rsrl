use std::f64;
use std::ops::Range;
use super::span::Span;

use rand::ThreadRng;
use rand::distributions::{Range as RngRange, IndependentSample};


// XXX: Address concern about [low, high) convention.

/// The basic dimension type.
pub trait Dimension {
    /// The corresponding primitive type.
    type Value;

    /// Sample a random value contained by this dimension.
    fn sample(&self, rng: &mut ThreadRng) -> Self::Value;

    /// Returns the total span of this dimension.
    fn span(&self) -> Span;

    /// Returns true iff `val` is contained within this dimension.
    fn contains(&self, val: &Self::Value) -> bool;
}

/// Dimension type with saturating upper/lower bounds.
pub trait BoundedDimension: Dimension
    where Self::Value: PartialOrd
{
    /// The upper/lower bound type; not necessarily equal to `Dimension::Value`.
    type ValueBound: PartialOrd;

    /// Returns a reference to the dimension's lower value bound (inclusive).
    fn lb(&self) -> &Self::ValueBound;

    /// Returns a reference to the dimension's upper value bound (inclusive).
    fn ub(&self) -> &Self::ValueBound;
}

/// Dimension type with bounds and a finite set of values.
pub trait FiniteDimension: BoundedDimension
    where Self::Value: PartialOrd
{
    /// Returns the finite range of values in this dimension.
    fn range(&self) -> Range<Self::Value>;
}


/// A null dimension.
#[derive(Clone, Copy)]
pub struct Null;

impl Dimension for Null {
    type Value = ();

    fn sample(&self, _: &mut ThreadRng) -> () {
        ()
    }

    fn span(&self) -> Span {
        Span::Null
    }

    fn contains(&self, _: &Self::Value) -> bool {
        false
    }
}


/// An infinite dimension.
#[derive(Clone, Copy)]
pub struct Infinite;

impl Dimension for Infinite {
    type Value = f64;

    fn sample(&self, _: &mut ThreadRng) -> f64 {
        unimplemented!()
    }

    fn span(&self) -> Span {
        Span::Infinite
    }

    fn contains(&self, _: &Self::Value) -> bool {
        true
    }
}


/// A continous dimensions.
#[derive(Clone, Copy)]
pub struct Continuous {
    lb: f64,
    ub: f64,
    range: RngRange<f64>,
}

impl Continuous {
    pub fn new(lb: f64, ub: f64) -> Continuous {
        Continuous {
            lb: lb,
            ub: ub,
            range: RngRange::new(lb, ub),
        }
    }
}

impl Dimension for Continuous {
    type Value = f64;

    fn sample(&self, rng: &mut ThreadRng) -> f64 {
        self.range.ind_sample(rng)
    }

    fn span(&self) -> Span {
        Span::Infinite
    }

    fn contains(&self, val: &Self::Value) -> bool {
        (val >= self.lb()) && (val < self.ub())
    }
}

impl BoundedDimension for Continuous {
    type ValueBound = Self::Value;

    fn lb(&self) -> &f64 {
        &self.lb
    }

    fn ub(&self) -> &f64 {
        &self.ub
    }
}


/// A finite discrete dimension.
#[derive(Clone)]
pub struct Discrete {
    lb: usize,
    ub: usize,
    range: RngRange<usize>,
}

impl Discrete {
    pub fn new(size: usize) -> Discrete {
        Discrete {
            lb: 0,
            ub: size - 1,
            range: RngRange::new(0, size),
        }
    }
}

impl Dimension for Discrete {
    type Value = usize;

    fn sample(&self, rng: &mut ThreadRng) -> usize {
        self.range.ind_sample(rng)
    }

    fn span(&self) -> Span {
        Span::Finite(self.ub + 1)
    }

    fn contains(&self, val: &Self::Value) -> bool {
        val < self.ub()
    }
}

impl BoundedDimension for Discrete {
    type ValueBound = usize;

    fn lb(&self) -> &usize {
        &self.lb
    }

    fn ub(&self) -> &usize {
        &self.ub
    }
}

impl FiniteDimension for Discrete {
    fn range(&self) -> Range<Self::Value> {
        0..(self.ub + 1)
    }
}


impl<D: Dimension> Dimension for Box<D> {
    type Value = D::Value;

    fn sample(&self, rng: &mut ThreadRng) -> D::Value {
        (**self).sample(rng)
    }

    fn span(&self) -> Span {
        (**self).span()
    }

    fn contains(&self, val: &Self::Value) -> bool {
        (**self).contains(val)
    }
}

impl<'a, D: Dimension> Dimension for &'a D {
    type Value = D::Value;

    fn sample(&self, rng: &mut ThreadRng) -> D::Value {
        (**self).sample(rng)
    }

    fn span(&self) -> Span {
        (**self).span()
    }

    fn contains(&self, val: &Self::Value) -> bool {
        (**self).contains(val)
    }
}

// TODO: Use quickcheck here to more extenisively test calls to contains...

#[cfg(test)]
mod tests {
    use super::{Dimension, BoundedDimension, FiniteDimension};
    use super::{Null, Infinite, Continuous, Discrete};

    use rand::thread_rng;
    use geometry::Span;

    #[test]
    fn test_null() {
        let d = Null;
        let mut rng = thread_rng();

        assert_eq!(d.sample(&mut rng), ());
        assert_eq!(d.span(), Span::Null);

        assert!(!d.contains(&()));
    }

    #[test]
    fn test_infinite() {
        let d = Infinite;

        assert_eq!(d.span(), Span::Infinite);

        assert!(d.contains(&123.456));
    }

    #[test]
    #[should_panic]
    fn test_infinite_sample() {
        let d = Infinite;
        let mut rng = thread_rng();

        let s = d.sample(&mut rng);
    }

    #[test]
    fn test_continuous() {
        for (lb, ub) in vec![(0.0, 5.0), (-5.0, 5.0), (-5.0, 0.0)] {
            let d = Continuous::new(lb, ub);
            let mut rng = thread_rng();

            assert_eq!(d.span(), Span::Infinite);

            assert!(!d.contains(&ub));
            assert!(d.contains(&lb));
            assert!(d.contains(&((lb + ub) / 2.0)));

            for _ in 0..100 {
                let s = d.sample(&mut rng);
                assert!(s < ub);
                assert!(s >= lb);
                assert!(d.contains(&s));
            }
        }
    }
}
