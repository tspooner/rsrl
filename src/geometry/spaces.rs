use super::Span;
use super::dimensions::{self, Dimension, Partitioned};
use rand::ThreadRng;
use std::fmt::Debug;
use std::iter::FromIterator;
use std::slice::Iter;


/// Trait for defining geometric spaces.
pub trait Space {
    /// The data representation of the space.
    type Repr: Debug + Clone;

    /// Generate a random sample from the space.
    fn sample(&self, rng: &mut ThreadRng) -> Self::Repr;

    /// Return the number of dimensions in the space.
    fn dim(&self) -> usize;

    /// Return the number of linear combinations of values in the space.
    fn span(&self) -> Span;
}

/// Geometric space for agent actions.
///
/// Currently the only supported representation for actions. In future we will need to handle
/// continuous actions.
pub type ActionSpace = UnitarySpace<dimensions::Discrete>;


/// An empty space.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct EmptySpace;

impl Space for EmptySpace {
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


/// 1-dimensional space.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct UnitarySpace<D: Dimension>(D);

impl<D: Dimension> UnitarySpace<D> {
    pub fn new(d: D) -> Self {
        UnitarySpace(d)
    }
}

impl UnitarySpace<dimensions::Continuous> {
    pub fn partitioned(self, density: usize) -> UnitarySpace<Partitioned> {
        UnitarySpace(Partitioned::from_continuous(self.0, density))
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


/// 2-dimensional homogeneous space.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct PairSpace<D1, D2>((D1, D2))
    where D1: Dimension,
          D2: Dimension;

impl<D1: Dimension, D2: Dimension> PairSpace<D1, D2> {
    pub fn new(d1: D1, d2: D2) -> Self {
        PairSpace((d1, d2))
    }
}

impl PairSpace<dimensions::Continuous, dimensions::Continuous> {
    pub fn partitioned(self, density: usize) -> PairSpace<Partitioned, Partitioned> {
        PairSpace((Partitioned::from_continuous((self.0).0, density),
                   Partitioned::from_continuous((self.0).1, density)))
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
        (self.0).0.span()*(self.0).1.span()
    }
}


/// N-dimensional homogeneous space.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct RegularSpace<D: Dimension> {
    dimensions: Vec<D>,
    span: Span,
}

impl<D: Dimension> RegularSpace<D> {
    pub fn new(dimensions: Vec<D>) -> Self {
        let mut s = Self::empty();

        for d in dimensions {
            s = s.push(d);
        }

        s
    }

    pub fn empty() -> Self {
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

impl RegularSpace<dimensions::Continuous> {
    pub fn partitioned(self, density: usize) -> RegularSpace<Partitioned> {
        self.into_iter()
            .map(|d| Partitioned::from_continuous(d, density))
            .collect()
    }
}

impl RegularSpace<dimensions::Partitioned> {
    pub fn centres(&self) -> Vec<Vec<f64>> {
        self.dimensions
            .iter()
            .map(|d| d.centres())
            .collect()
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

impl<D: Dimension> FromIterator<D> for RegularSpace<D> {
    fn from_iter<I: IntoIterator<Item = D>>(iter: I) -> Self {
        let mut s = Self::empty();

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


#[cfg(test)]
mod tests {
    use super::{Space, EmptySpace, UnitarySpace, PairSpace, RegularSpace};
    use geometry::Span;
    use geometry::dimensions::*;
    use ndarray::arr1;
    use rand::thread_rng;

    #[test]
    fn test_null_space() {
        let ns = EmptySpace;
        let mut rng = thread_rng();

        assert_eq!(ns.sample(&mut rng), ());
        assert_eq!(ns.dim(), 0);
        assert_eq!(ns.span(), Span::Null);
    }

    #[test]
    fn test_unitary_space() {
        let us = UnitarySpace::new(Discrete::new(2));
        let mut rng = thread_rng();

        let mut counts = arr1(&vec![0.0; 2]);
        for _ in 0..5000 {
            let sample = us.sample(&mut rng);
            counts[sample] += 1.0;

            assert!(sample == 0 || sample == 1);
        }

        assert!((counts/5000.0).all_close(&arr1(&vec![0.5; 2]), 1e-1));
        assert_eq!(us.dim(), 1);
        assert_eq!(us.span(), Span::Finite(2));
    }

    #[test]
    fn test_pair_space() {
        let ps = PairSpace::new(Discrete::new(2), Discrete::new(2));

        let mut rng = thread_rng();

        let mut c1 = arr1(&vec![0.0; 2]);
        let mut c2 = arr1(&vec![0.0; 2]);
        for _ in 0..5000 {
            let sample = ps.sample(&mut rng);

            c1[sample.0] += 1.0;
            c2[sample.1] += 1.0;

            assert!(sample.0 == 0 || sample.0 == 1);
            assert!(sample.1 == 0 || sample.1 == 1);
        }

        assert!((c1/5000.0).all_close(&arr1(&vec![0.5; 2]), 1e-1));
        assert!((c2/5000.0).all_close(&arr1(&vec![0.5; 2]), 1e-1));

        assert_eq!(ps.dim(), 2);
        assert_eq!(ps.span(), Span::Finite(4));
    }

    #[test]
    fn test_regular_space() {
        let space = RegularSpace::new(vec![Discrete::new(2); 2]);

        let mut rng = thread_rng();

        let mut c1 = arr1(&vec![0.0; 2]);
        let mut c2 = arr1(&vec![0.0; 2]);
        for _ in 0..5000 {
            let sample = space.sample(&mut rng);

            c1[sample[0]] += 1.0;
            c2[sample[1]] += 1.0;

            assert!(sample[0] == 0 || sample[0] == 1);
            assert!(sample[1] == 0 || sample[1] == 1);
        }

        assert!((c1/5000.0).all_close(&arr1(&vec![0.5; 2]), 1e-1));
        assert!((c2/5000.0).all_close(&arr1(&vec![0.5; 2]), 1e-1));

        assert_eq!(space.dim(), 2);
        assert_eq!(space.span(), Span::Finite(4));
    }
}
