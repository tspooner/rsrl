use super::{Function, Parameterised, Linear, QFunction};

use utils::dot;
use utils::kernels::Kernel;
use ndarray::{Array1, Array2, ArrayView};
use geometry::{RegularSpace};
use geometry::dimensions::Continuous;


/// Represents the location and kernel associated with a basis function.
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
}


/// Generic basis function network for function representation.
pub struct BasisNetwork {
    bases: Vec<BasisFunction>,
    weights: Array2<f64>,
}

impl BasisNetwork {
    pub fn new(bases: Vec<BasisFunction>, n_outputs: usize) -> Self
    {
        let n_features = bases.len();

        BasisNetwork {
            bases: bases,
            weights: Array2::<f64>::zeros((n_features, n_outputs)),
        }
    }
}

impl Function<[f64], f64> for BasisNetwork {
    fn evaluate(&self, inputs: &[f64]) -> f64 {
        // Compute the feature vector phi:
        let phi = self.phi(inputs);

        // Apply matrix multiplication and return Vec<f64>:
        dot(self.weights.column(0).as_slice().unwrap(),
            phi.as_slice().unwrap())
    }
}

impl Function<[f64], Vec<f64>> for BasisNetwork
{
    fn evaluate(&self, input: &[f64]) -> Vec<f64> {
        // Compute the feature vector phi:
        let phi = self.phi(input);

        // Apply matrix multiplication and return Vec<f64>:
        (self.weights.t().dot(&phi)).into_raw_vec()
    }
}

add_vec_support!(BasisNetwork, Function, f64, Vec<f64>);


impl Parameterised<[f64], f64> for BasisNetwork {
    fn update(&mut self, inputs: &[f64], error: f64) {
        // Compute the feature vector phi:
        let phi = self.phi(inputs);

        // Update the weights via addassign:
        self.weights.scaled_add(error, &phi);
    }
}

impl Parameterised<[f64], Vec<f64>> for BasisNetwork
{
    fn update(&mut self, input: &[f64], errors: Vec<f64>) {
        // Compute the feature vector phi:
        let phi = self.phi(input).into_shape((self.bases.len(), 1)).unwrap();

        // Compute update matrix using phi and column-wise errors:
        let update_matrix =
            ArrayView::from_shape((1, self.weights.cols()), errors.as_slice()).unwrap();

        // Update the weights via addassign:
        self.weights += &phi.dot(&update_matrix)
    }
}

add_vec_support!(BasisNetwork, Parameterised, f64, Vec<f64>);


impl Linear<[f64]> for BasisNetwork {
    fn phi(&self, input: &[f64]) -> Array1<f64> {
        Array1::from_shape_fn((self.bases.len(),), |i| {
            self.bases[i].evaluate(input)
        })
    }
}

add_vec_support!(BasisNetwork, Linear);


impl QFunction<RegularSpace<Continuous>> for BasisNetwork
{
    fn evaluate_action(&self, input: &Vec<f64>, action: usize) -> f64 {
        // Apply matrix multiplication and return f64:
        dot(self.weights.column(action).as_slice().unwrap(),
            self.phi(input).as_slice().unwrap())
    }

    fn update_action(&mut self, input: &Vec<f64>, action: usize, error: f64) {
        // Compute the feature vector phi:
        let phi = self.phi(input);

        // Update the weights via scaled_add:
        self.weights.column_mut(action).scaled_add(error, &phi);
    }
}
