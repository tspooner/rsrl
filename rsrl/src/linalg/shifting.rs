use crate::{basis::Projector, features::Features};

/// Shift the output of a `Projector` instance by some fixed amount.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Shift<P> {
    projector: P,
    offset: f64,
}

impl<P> Shift<P> {
    pub fn new(projector: P, offset: f64) -> Self {
        Shift {
            projector: projector,
            offset: offset,
        }
    }
}

impl<P: Projector> Projector for Shift<P> {
    type Features = P::Features;

    fn n_features(&self) -> usize {
        self.projector.n_features()
    }

    fn project(&self, input: &[f64]) -> Self::Features {
        let mut f = self.projector.project(input);

        f.map_inplace(|a| a + self.offset);

        f
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        basis::fixed::Constant,
        features::Features,
        geometry::Vector,
    };
    use quickcheck::quickcheck;
    use super::*;

    #[test]
    fn test_shifting() {
        fn prop_output(length: usize, v1: f64, v2: f64) -> bool {
            let p = Shift::new(Constant::new(length, v1), v2);
            let f: Vector<_> = p.project(&[0.0]).into();

            f.into_iter().all(|&v| (v - (v1 + v2)) < 1e-7)
        }

        quickcheck(prop_output as fn(usize, f64, f64) -> bool);
    }
}
