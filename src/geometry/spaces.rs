use std::iter::FromIterator;
use std::slice::Iter;
use rand::ThreadRng;

use super::Span;
use super::dimensions;
use super::dimensions::{Dimension, Partitioned};


pub trait Space {
    type Repr;

    fn sample(&self, rng: &mut ThreadRng) -> Self::Repr;

    fn dim(&self) -> usize;
    fn span(&self) -> Span;
}

pub type ActionSpace = UnitarySpace<dimensions::Discrete>;


pub struct NullSpace;

impl Space for NullSpace {
    type Repr = ();

    fn sample(&self, _: &mut ThreadRng) -> Self::Repr {
        ()
    }

    fn dim(&self) -> usize {
        0
    }

    fn span(&self) -> Span {
        Span::Null
    }
}


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
    pub fn partitioned(self, density: usize) -> PairSpace<Partitioned, Partitioned> {
        PairSpace((Partitioned::from_continuous((self.0).0, density),
                   Partitioned::from_continuous((self.0).1, density)))
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
    pub fn partitioned(self, density: usize) -> RegularSpace<Partitioned> {
        self.into_iter()
            .map(|d| Partitioned::from_continuous(d, density))
            .collect()
    }
}

impl RegularSpace<dimensions::Partitioned> {
    pub fn centres(&self) -> Vec<Vec<f64>> {
        self.dimensions.iter()
            .map(|d| d.centres())
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


// pub struct HeterogeneousSpace {
    // dimensions: Vec<Dimension>,
    // span: Span
// }

// impl HeterogeneousSpace {
    // pub fn new() -> Self {
     // HeterogeneousSpace {
            // dimensions: vec![],
            // span: Span::Null
        // }
    // }

    // pub fn push(mut self, d: D) -> Self {
        // self.span = self.span * d.span();
        // self.dimensions.push(d);
        // self
    // }

    // pub fn iter(&self) -> Iter<Dimension> {
        // self.dimensions.iter()
    // }
// }

// impl<D: Dimension> Space for HeterogeneousSpace<D> {
    // type Repr = Vec<D::Value>;

    // fn sample(&self, rng: &mut ThreadRng) -> Self::Repr {
        // self.dimensions.iter().map(|d| d.sample(rng)).collect()
    // }

    // fn dim(&self) -> usize {
        // self.dimensions.len()
    // }

    // fn span(&self) -> Span {
        // self.span
    // }
// }

    // fn sample(&self, rng: &mut ThreadRng) -> Self::Repr;

    // fn dim(&self) -> usize;
    // fn span(&self) -> Span;

#[cfg(test)]
mod tests {
    use super::{Space, NullSpace, UnitarySpace, PairSpace, RegularSpace};

    use rand::thread_rng;
    use geometry::Span;
    use geometry::dimensions::*;

    #[test]
    fn test_null_space() {
        let ns = NullSpace;
        let mut rng = thread_rng();

        assert_eq!(ns.sample(&mut rng), ());
        assert_eq!(ns.dim(), 0);
        assert_eq!(ns.span(), Span::Null);
    }
}
