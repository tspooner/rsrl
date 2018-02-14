use super::{Projection, Projector};
use Vector;
use fa::Function;
use geometry::RegularSpace;
use geometry::dimensions::Continuous;
use geometry::kernels::Kernel;

/// Represents the location and kernel associated with a basis function.
pub struct BasisFunction {
    pub loc: Vec<f64>,
    pub kernel: Box<Kernel>,
}

impl BasisFunction {
    pub fn new(loc: Vec<f64>, kernel: Box<Kernel>) -> Self {
        BasisFunction {
            loc: loc,
            kernel: kernel,
        }
    }
}

impl Function<[f64], f64> for BasisFunction {
    fn evaluate(&self, input: &[f64]) -> f64 { self.kernel.kernel(&self.loc, input) }
}

/// Heterogeneous basis network projector for arbitrary combinations of basis
/// functions.
pub struct BasisNetwork {
    bases: Vec<BasisFunction>,
}

impl BasisNetwork {
    pub fn new(bases: Vec<BasisFunction>) -> Self { BasisNetwork { bases: bases } }
}

impl Projector<RegularSpace<Continuous>> for BasisNetwork {
    fn project(&self, input: &Vec<f64>) -> Projection {
        let phi = Vector::from_shape_fn((self.bases.len(),), |i| self.bases[i].evaluate(input));

        Projection::Dense(phi)
    }

    fn dim(&self) -> usize { unimplemented!() }

    fn size(&self) -> usize { self.bases.len() }

    fn activation(&self) -> usize { self.size() }

    fn equivalent(&self, _: &Self) -> bool { unimplemented!() }
}

#[cfg(test)]
mod tests {
    use super::BasisFunction;
    use fa::Function;
    use geometry::kernels::{Exponential, Kernel, SquaredExp};

    #[test]
    fn test_consistency_exp() {
        let loc = vec![0.0];

        let k = Exponential::new(2.0, 1.0);
        let f = BasisFunction::new(loc.clone(), Box::new(k));

        assert_eq!(f.evaluate(&vec![1.0]), k.kernel(&loc, &vec![1.0]));
        assert_eq!(f.evaluate(&vec![2.0]), k.kernel(&loc, &vec![2.0]));
        assert_eq!(f.evaluate(&vec![-1.0]), k.kernel(&loc, &vec![-1.0]));
        assert_eq!(f.evaluate(&vec![-2.0]), k.kernel(&loc, &vec![-2.0]));
    }

    #[test]
    fn test_consistency_sq_exp() {
        let loc = vec![0.0];

        let k = SquaredExp::new(2.0, 1.0);
        let f = BasisFunction::new(loc.clone(), Box::new(k));

        assert_eq!(f.evaluate(&vec![1.0]), k.kernel(&loc, &vec![1.0]));
        assert_eq!(f.evaluate(&vec![2.0]), k.kernel(&loc, &vec![2.0]));
        assert_eq!(f.evaluate(&vec![-1.0]), k.kernel(&loc, &vec![-1.0]));
        assert_eq!(f.evaluate(&vec![-2.0]), k.kernel(&loc, &vec![-2.0]));
    }
}
