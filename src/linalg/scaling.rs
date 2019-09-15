use crate::{basis::Projector, features::Features};

/// Scale the output of a `Projector` instance by some fixed amount.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Scale<P> {
    projector: P,
    scale: f64,
}

impl<P> Scale<P> {
    pub fn new(projector: P, scale: f64) -> Self {
        Scale {
            projector: projector,
            scale: scale,
        }
    }
}

impl<P: Projector> Projector for Scale<P> {
    type Features = P::Features;

    fn n_features(&self) -> usize {
        self.projector.n_features()
    }

    fn project(&self, input: &[f64]) -> Self::Features {
        let mut p = self.projector.project(input);

        p.map_inplace(|a| a * self.scale);

        p
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
    fn test_scaling() {
        fn prop_output(length: usize, v1: f64, v2: f64) -> bool {
            let p = Scale::new(Constant::new(length, v1), v2);
            let f: Vector<_> = p.project(&[0.0]).into();

            f.into_iter().all(|&v| (v - v1 * v2) < 1e-7)
        }

        quickcheck(prop_output as fn(usize, f64, f64) -> bool);
    }
}
