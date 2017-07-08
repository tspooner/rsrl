use std::f64;
use std::fmt;
use std::ops::Range;
use super::span::Span;

use rand::ThreadRng;
use rand::distributions::{Range as RngRange, IndependentSample};


// XXX: Address concern about [low, high) convention.

/// The basic dimension type.
pub trait Dimension
{
    /// The corresponding primitive type.
    type Value: Clone;

    /// Sample a random value contained by this dimension.
    fn sample(&self, rng: &mut ThreadRng) -> Self::Value;

    /// Map a compatible input into a valid value of this dimension.
    fn convert(&self, val: f64) -> Self::Value;

    /// Returns the total span of this dimension.
    fn span(&self) -> Span;
}

/// Dimension type with saturating upper/lower bounds.
pub trait BoundedDimension: Dimension
    where Self::Value: PartialOrd
{
    /// The upper/lower bound type; not necessarily equal to `Dimension::Value`.
    type ValueBound: PartialOrd;

    /// Returns a reference to the dimension's lower value bound (inclusive).
    fn lb(&self) -> &Self::ValueBound;

    /// Returns a reference to the dimension's upper value bound (exclusive).
    fn ub(&self) -> &Self::ValueBound;

    /// Returns true iff `val` is within the dimension's bounds.
    fn contains(&self, val: &Self::ValueBound) -> bool;
}

/// Dimension type with bounds and a finite set of values.
pub trait FiniteDimension: BoundedDimension
    where Self::Value: PartialOrd
{
    /// Returns the finite range of values in this dimension.
    fn range(&self) -> Range<Self::Value>;
}


/// A null dimension.
#[derive(Clone, Copy, Debug)]
pub struct Null;

impl Dimension for Null {
    type Value = ();

    fn sample(&self, _: &mut ThreadRng) -> () {
        ()
    }

    fn convert(&self, _: f64) -> Self::Value { () }

    fn span(&self) -> Span {
        Span::Null
    }
}


/// An infinite dimension.
#[derive(Clone, Copy, Debug)]
pub struct Infinite;

impl Infinite {
    pub fn bounded(self, lb: f64, ub: f64) -> Continuous {
        Continuous::new(lb, ub)
    }
}

impl Dimension for Infinite {
    type Value = f64;

    fn sample(&self, _: &mut ThreadRng) -> f64 {
        unimplemented!()
    }

    fn convert(&self, val: f64) -> Self::Value { val }

    fn span(&self) -> Span {
        Span::Infinite
    }
}


/// A continous dimension.
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

    fn convert(&self, val: f64) -> Self::Value {
        clip!(self.lb, val, self.ub)
    }

    fn span(&self) -> Span {
        Span::Infinite
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

    fn contains(&self, val: &Self::ValueBound) -> bool {
        (val >= self.lb()) && (val < self.ub())
    }
}

impl fmt::Debug for Continuous {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Continuous")
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .finish()
    }
}


/// A finite, uniformly partitioned continous dimension.
#[derive(Clone, Copy)]
pub struct Partitioned {
    lb: f64,
    ub: f64,
    density: usize,

    range: RngRange<f64>,
}

impl Partitioned {
    pub fn new(lb: f64, ub: f64, density: usize) -> Partitioned {
        Partitioned {
            lb: lb,
            ub: ub,
            density: density,

            range: RngRange::new(lb, ub),
        }
    }

    pub fn from_continuous(d: Continuous, density: usize) -> Partitioned {
        Partitioned {
            lb: d.lb,
            ub: d.ub,
            density: density,

            range: d.range,
        }
    }

    pub fn to_partition(&self, val: f64) -> usize {
        let clipped = clip!(self.lb, val, self.ub);

        let diff = clipped - self.lb;
        let range = self.ub - self.lb;

        let i = ((self.density as f64) * diff / range).floor() as usize;

        if i == self.density { i - 1 } else { i }
    }

    pub fn centres(&self) -> Vec<f64> {
        let w = (self.ub - self.lb) / self.density as f64;
        let hw = w / 2.0;

        (0..self.density).map(|i| self.lb + w*(i as f64) - hw).collect()
    }

    pub fn partition_width(&self) -> f64 {
        (self.lb - self.ub) / self.density as f64
    }

    pub fn density(&self) -> usize {
        self.density
    }
}

impl Dimension for Partitioned {
    type Value = usize;

    fn sample(&self, rng: &mut ThreadRng) -> usize {
        self.to_partition(self.range.ind_sample(rng))
    }

    fn convert(&self, val: f64) -> Self::Value {
        self.to_partition(val)
    }

    fn span(&self) -> Span {
        Span::Finite(self.density)
    }
}

impl BoundedDimension for Partitioned {
    type ValueBound = f64;

    fn lb(&self) -> &f64 {
        &self.lb
    }

    fn ub(&self) -> &f64 {
        &self.ub
    }

    fn contains(&self, val: &Self::ValueBound) -> bool {
        (val >= self.lb()) && (val < self.ub())
    }
}

impl FiniteDimension for Partitioned {
    fn range(&self) -> Range<Self::Value> {
        0..(self.density + 1)
    }
}

impl fmt::Debug for Partitioned {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Partitioned")
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .field("density", &self.density)
            .finish()
    }
}


/// A finite discrete dimension.
#[derive(Clone, Copy)]
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

    fn convert(&self, val: f64) -> Self::Value { val as usize }

    fn span(&self) -> Span {
        Span::Finite(self.ub + 1)
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

    fn contains(&self, val: &Self::Value) -> bool {
        *val < self.ub
    }
}

impl FiniteDimension for Discrete {
    fn range(&self) -> Range<Self::Value> {
        0..(self.ub + 1)
    }
}

impl fmt::Debug for Discrete {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Discrete")
            .field("lb", &self.lb)
            .field("ub", &self.ub)
            .finish()
    }
}


impl<D: Dimension> Dimension for Box<D> {
    type Value = D::Value;

    fn sample(&self, rng: &mut ThreadRng) -> Self::Value {
        (**self).sample(rng)
    }

    fn convert(&self, val: f64) -> Self::Value {
        (**self).convert(val)
    }

    fn span(&self) -> Span {
        (**self).span()
    }
}

impl<'a, D: Dimension> Dimension for &'a D {
    type Value = D::Value;

    fn sample(&self, rng: &mut ThreadRng) -> Self::Value {
        (**self).sample(rng)
    }

    fn convert(&self, val: f64) -> Self::Value {
        (**self).convert(val)
    }

    fn span(&self) -> Span {
        (**self).span()
    }
}

// TODO: Use quickcheck here to more extenisively test calls to contains...

#[cfg(test)]
mod tests {
    use super::{Dimension, BoundedDimension, FiniteDimension};
    use super::{Null, Infinite, Continuous, Partitioned, Discrete};

    use rand::thread_rng;
    use geometry::Span;

    #[test]
    fn test_null() {
        let d = Null;
        let mut rng = thread_rng();

        assert_eq!(d.sample(&mut rng), ());
        assert_eq!(d.span(), Span::Null);
    }

    #[test]
    fn test_infinite() {
        let d = Infinite;

        assert_eq!(d.span(), Span::Infinite);
    }

    #[test]
    #[should_panic]
    fn test_infinite_sample() {
        let d = Infinite;
        let mut rng = thread_rng();

        let _ = d.sample(&mut rng);
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

    #[test]
    fn test_partitioned() {
        for (lb, ub, density) in vec![(0.0, 5.0, 5), (-5.0, 5.0, 10), (-5.0, 0.0, 5)] {
            let d = Partitioned::new(lb, ub, density);
            let mut rng = thread_rng();

            assert_eq!(d.span(), Span::Finite(density));

            assert!(!d.contains(&ub));
            assert!(d.contains(&lb));
            assert!(d.contains(&((lb + ub) / 2.0)));

            for _ in 0..100 {
                let s = d.sample(&mut rng);
                assert!(s < density);
                assert!(s >= 0);
            }
        }
    }

    #[test]
    fn test_discrete() {
        for size in vec![5, 10, 100] {
            let d = Discrete::new(size);
            let mut rng = thread_rng();

            assert_eq!(d.span(), Span::Finite(size));

            assert!(!d.contains(&(size - 1)));
            assert!(d.contains(&0));
            assert!(d.contains(&(size - 2)));

            for _ in 0..100 {
                let s = d.sample(&mut rng);
                assert!(s < size);
                assert!(s >= 0);
            }
        }
    }
}
