use super::{Function, Parameterised, Linear, QFunction};

use utils::{cartesian_product, dot};
use ndarray::{Axis, ArrayView, Array1, Array2};
use geometry::{Span, Space, RegularSpace};
use geometry::dimensions::{Partition, Continuous};


/// Optimised radial basis function network for function representation.
pub struct RBFNetwork {
    mu: Array2<f64>,
    gamma: Array1<f64>,

    weights: Array2<f64>,
}

impl RBFNetwork
{
    pub fn new(input_space: RegularSpace<Partition>, n_outputs: usize) -> Self
    {
        let n_features = match input_space.span() {
            Span::Finite(s) => s,
            _ =>
                panic!("`RBFNetwork` function approximator only supports \
                        finite input spaces.")
        };

        let centres = input_space.compute_centres();
        let flat_combs =
            cartesian_product(&centres)
            .iter().cloned()
            .flat_map(|e| e).collect();

        RBFNetwork {
            mu: Array2::from_shape_vec((n_features, input_space.dim()), flat_combs).unwrap(),
            gamma: input_space.iter().map(|d| {
                let s = d.partition_width();
                -1.0 / (s * s)
            }).collect(),

            weights: Array2::<f64>::zeros((n_features, n_outputs)),
        }
    }
}


impl Function<[f64], f64> for RBFNetwork
{
    fn evaluate(&self, input: &[f64]) -> f64 {
        // Compute the feature vector phi:
        let phi = self.phi(input);

        // Apply matrix multiplication and return f64:
        dot(self.weights.column(0).as_slice().unwrap(),
            phi.as_slice().unwrap())
    }
}

impl Function<[f64], Vec<f64>> for RBFNetwork
{
    fn evaluate(&self, input: &[f64]) -> Vec<f64> {
        // Compute the feature vector phi:
        let phi = self.phi(input);

        // Apply matrix multiplication and return Vec<f64>:
        (self.weights.t().dot(&phi)).into_raw_vec()
    }
}

add_vec_support!(RBFNetwork, Function, f64, Vec<f64>);


impl Parameterised<[f64], f64> for RBFNetwork
{
    fn update(&mut self, input: &[f64], error: f64) {
        // Compute the feature vector phi:
        let phi = self.phi(input);

        // Update the weights via scaled_add:
        self.weights.scaled_add(error, &phi);
    }
}

impl Parameterised<[f64], Vec<f64>> for RBFNetwork
{
    fn update(&mut self, input: &[f64], errors: Vec<f64>) {
        // Compute the feature vector phi:
        let phi = self.phi(input).into_shape((self.weights.rows(), 1)).unwrap();

        // Compute update matrix using phi and column-wise errors:
        let update_matrix =
            ArrayView::from_shape((1, self.weights.cols()), errors.as_slice()).unwrap();

        // Update the weights via addassign:
        self.weights += &phi.dot(&update_matrix)
    }
}

add_vec_support!(RBFNetwork, Parameterised, f64, Vec<f64>);


impl Linear<[f64]> for RBFNetwork
{
    fn phi(&self, input: &[f64]) -> Array1<f64> {
        let d = &self.mu - &ArrayView::from_shape((1, self.mu.cols()), input).unwrap();
        let e = (&d * &d * &self.gamma).mapv(|v| v.exp()).sum(Axis(1));
        let z = e.sum(Axis(0));

        e / z
    }
}

add_vec_support!(RBFNetwork, Linear);


impl QFunction<RegularSpace<Continuous>> for RBFNetwork
{
    fn evaluate_action(&self, input: &Vec<f64>, action: usize) -> f64 {
        // Compute the feature vector phi:
        let phi = self.phi(input);

        // Apply matrix multiplication and return f64:
        dot(self.weights.column(action).as_slice().unwrap(),
            phi.as_slice().unwrap())
    }

    fn update_action(&mut self, input: &Vec<f64>, action: usize, error: f64) {
        // Compute the feature vector phi:
        let phi = self.phi(input);

        // Update the weights via scaled_add:
        self.weights.column_mut(action).scaled_add(error, &phi);
    }
}
