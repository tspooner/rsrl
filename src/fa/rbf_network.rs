use super::{Function, Parameterised};

use utils::{cartesian_product, dot};
use ndarray::{Axis, ArrayView, Array1, Array2};
use geometry::{Span, Space, RegularSpace};
use geometry::dimensions::Partition;


pub struct RBFNetwork {
    mu: Array2<f64>,
    gamma: Array1<f64>,
    weights: Array2<f64>,
}

impl RBFNetwork {
    pub fn new(input_space: RegularSpace<Partition>,
               n_outputs: usize) -> Self
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

    fn phi(&self, inputs: &[f64]) -> Array1<f64> {
        let d = &self.mu - &ArrayView::from_shape((1, self.mu.cols()), inputs).unwrap();
        let e = (&d * &d * &self.gamma).mapv(|v| v.exp()).sum(Axis(1));
        let z = e.sum(Axis(0));

        e / z
    }
}


impl Function<[f64], f64> for RBFNetwork {
    fn evaluate(&self, inputs: &[f64]) -> f64 {
        // Compute the feature vector phi:
        let phi = self.phi(inputs);

        // Apply matrix multiplication and return f64:
        dot(self.weights.column(0).into_slice().unwrap(),
            phi.as_slice().unwrap())
    }

    fn n_outputs(&self) -> usize {
        1
    }
}

impl Function<[f64], Vec<f64>> for RBFNetwork {
    fn evaluate(&self, inputs: &[f64]) -> Vec<f64> {
        // Compute the feature vector phi:
        let phi = self.phi(inputs);

        // Apply matrix multiplication and return Vec<f64>:
        (self.weights.t().dot(&phi)).into_raw_vec()
    }

    fn n_outputs(&self) -> usize {
        self.weights.cols()
    }
}

add_support!(RBFNetwork, Function, Vec<f64>, [f64, Vec<f64>]);


impl Parameterised<[f64], f64> for RBFNetwork {
    fn update(&mut self, inputs: &[f64], error: &f64) {
        // Compute the feature vector phi:
        let phi = self.phi(inputs);

        // Update the weights via scaled_add:
        self.weights.column_mut(0).scaled_add(*error, &phi);
    }
}

impl Parameterised<[f64], [f64]> for RBFNetwork {
    fn update(&mut self, inputs: &[f64], errors: &[f64]) {
        // Compute the feature vector phi:
        let phi = self.phi(inputs).into_shape((1, self.weights.rows())).unwrap();

        // Compute update matrix using phi and column-wise errors:
        let update_matrix =
            ArrayView::from_shape((1, self.weights.cols()), errors).unwrap();

        // Update the weights via addassign:
        self.weights += &phi.t().dot(&update_matrix)
    }
}

add_support!(RBFNetwork, Parameterised, Vec<f64>, [f64, [f64]]);
