use std::iter::FromIterator;
use std::slice::Iter;
use rand::ThreadRng;

use super::Span;
use super::dimensions;
use super::dimensions::{Dimension, BoundedDimension, Discrete};
use super::partitioning::Partitions;


pub trait Space {
    type Repr;

    fn sample(&self, rng: &mut ThreadRng) -> Self::Repr;

    fn dim(&self) -> usize;
    fn span(&self) -> Span;
}

pub type ActionSpace = UnitarySpace<Discrete>;


pub struct UnitarySpace<D: Dimension>(D);

impl<D: Dimension> UnitarySpace<D> {
    pub fn new(d: D) -> Self {
        UnitarySpace(d)
    }
}

impl<D: Dimension> Space for UnitarySpace<D> {
    type Repr = D::Value;

    fn sample(&self, rng: &mut ThreadRng) -> Self::Repr {
        self.0.sample(rng)
    }

    fn dim(&self) -> usize {
        1
    }

    fn span(&self) -> Span {
        self.0.span()
    }
}


pub struct RegularSpace<D: Dimension> {
    dimensions: Vec<D>,
    span: Span,
}

impl<D: Dimension> RegularSpace<D> {
    pub fn new() -> Self {
        RegularSpace {
            dimensions: vec![],
            span: Span::Null,
        }
    }

    pub fn push(mut self, d: D) -> Self {
        self.span = self.span * d.span();
        self.dimensions.push(d);
        self
    }

    pub fn iter(&self) -> Iter<D> {
        self.dimensions.iter()
    }
}

impl<D: Dimension> Space for RegularSpace<D> {
    type Repr = Vec<D::Value>;

    fn sample(&self, rng: &mut ThreadRng) -> Self::Repr {
        self.dimensions.iter().map(|d| d.sample(rng)).collect()
    }

    fn dim(&self) -> usize {
        self.dimensions.len()
    }

    fn span(&self) -> Span {
        self.span
    }
}

impl RegularSpace<dimensions::Continuous> {
    pub fn partitioned(&self, density: usize) -> Vec<Partitions> {
        self.iter()
            .map(|d| Partitions::new(*d.lb(), *d.ub(), density))
            .collect()
    }
}

impl<D: Dimension> FromIterator<D> for RegularSpace<D> {
    fn from_iter<I: IntoIterator<Item = D>>(iter: I) -> Self {
        let mut s = Self::new();

        for i in iter {
            s = s.push(i);
        }

        s
    }
}

impl<D: Dimension> IntoIterator for RegularSpace<D> {
    type Item = D;
    type IntoIter = ::std::vec::IntoIter<D>;

    fn into_iter(self) -> Self::IntoIter {
        self.dimensions.into_iter()
    }
}


// TODO: Bring back support for Null/Pair spaces.
// pub struct NullSpace(dimensions::Null);

// impl Space for NullSpace {
// type Repr = ();

// fn dim(&self) -> usize {
// 0
// }

// fn span(&self) -> Span {
// Span::Null
// }
// }


pub struct PairSpace<D1, D2>((D1, D2))
    where D1: Dimension,
          D2: Dimension;

impl<D1: Dimension, D2: Dimension> PairSpace<D1, D2> {
    pub fn new(d1: D1, d2: D2) -> Self {
        PairSpace((d1, d2))
    }
}

impl<D1: Dimension, D2: Dimension> Space for PairSpace<D1, D2> {
    type Repr = (D1::Value, D2::Value);

    fn sample(&self, rng: &mut ThreadRng) -> Self::Repr {
        ((self.0).0.sample(rng), (self.0).1.sample(rng))
    }

    fn dim(&self) -> usize {
        2
    }

    fn span(&self) -> Span {
        (self.0).0.span() * (self.0).1.span()
    }
}

impl PairSpace<dimensions::Continuous, dimensions::Continuous> {
    pub fn partitioned(&self, density: usize) -> (Partitions, Partitions) {
        (Partitions::new(*(self.0).0.lb(), *(self.0).0.ub(), density),
         Partitions::new(*(self.0).0.lb(), *(self.0).1.ub(), density))
    }
}


// pub struct MultiSpace<D: Dimension> {
// dimensions: Vec<D>,
// span: Span
// }

// impl<D: Dimension> MultiSpace<D> {
// pub fn new() -> Self {
// MultiSpace {
// dimensions: vec![],
// span: Span::Null
// }
// }

// pub fn push(mut self, d: D) -> Self {
// self.span = self.span * d.span();
// self.dimensions.push(d);
// self
// }

// pub fn iter(&self) -> Iter<D> {
// self.dimensions.iter()
// }
// }

// impl<D: Dimension> Space for MultiSpace<D> {
// type Repr = Vec<D::Value>;

// fn dim(&self) -> usize {
// self.dimensions.len()
// }

// fn span(&self) -> Span {
// self.span
// }
// }

// impl MultiSpace<dimensions::Continuous> {
// pub fn with_partitions(self, density: usize) -> MultiSpace<Partitioned> {
// self.into_iter().map(
// |d| Partitioned::from_continuous(d, density)).collect()
// }
// }
