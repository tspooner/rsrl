use fa::Function;
use super::Projection;
use ndarray::Array1;
use geometry::RegularSpace;
use geometry::kernels::Kernel;
use geometry::dimensions::Continuous;


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
    fn evaluate(&self, input: &[f64]) -> f64 {
        self.kernel.apply(&self.loc, input)
    }
}


pub struct BasisNetwork {
    bases: Vec<BasisFunction>,
}

impl BasisNetwork {
    pub fn new(bases: Vec<BasisFunction>) -> Self {
        BasisNetwork {
            bases: bases,
        }
    }
}

impl Projection<RegularSpace<Continuous>> for BasisNetwork {
    fn project(&self, input: &Vec<f64>) -> Array1<f64> {
        Array1::from_shape_fn((self.bases.len(),), |i| {
            self.bases[i].evaluate(input)
        })
    }

    fn dim(&self) -> usize {
        self.bases.len()
    }

    fn equivalent(&self, _: &Self) -> bool {
        unimplemented!()
    }
}
