use super::{Function, Parameterised};

use std::iter::FromIterator;
use utils::kernels::Kernel;
use ndarray::{ArrayView, Array1, Array2};


pub struct BasisFunction {
    loc: Vec<f64>,
    kernel: Box<Kernel>,
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

    fn n_outputs(&self) -> usize {
        1
    }
}


pub struct BasisNetwork {
    bases: Vec<BasisFunction>,
    weights: Array2<f64>,
}

impl BasisNetwork {
    pub fn new(bases: Vec<BasisFunction>,
               n_outputs: usize) -> Self
    {
        let n_features = bases.len();

        BasisNetwork {
            bases: bases,
            weights: Array2::<f64>::zeros((n_features, n_outputs)),
        }
    }

    fn phi<O: FromIterator<f64>>(&self, inputs: &[f64]) -> O {
        self.bases.iter().map(|b| b.evaluate(inputs)).collect()
    }
}

impl Function<[f64], Vec<f64>> for BasisNetwork {
    fn evaluate(&self, inputs: &[f64]) -> Vec<f64> {
        // Compute the feature vector phi:
        let phi: Array1<f64> = self.phi(inputs);

        // Apply matrix multiplication and return Vec<f64>:
        (self.weights.t().dot(&phi)).into_raw_vec()
    }

    fn n_outputs(&self) -> usize {
        self.weights.cols()
    }
}


impl Parameterised<[f64], [f64]> for BasisNetwork {
    fn update(&mut self, inputs: &[f64], errors: &[f64]) {
        // Compute the feature vector phi:
        let phi = self.phi::<Array1<f64>>(inputs)
            .into_shape((self.bases.len(), 1)).unwrap();

        // Compute update matrix using phi and column-wise errors:
        let update_matrix =
            ArrayView::from_shape((1, self.weights.cols()), errors).unwrap();

        // Update the weights via addassign:
        self.weights += &phi.dot(&update_matrix)
    }
}
