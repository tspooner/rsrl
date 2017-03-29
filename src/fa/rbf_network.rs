use {Function, Parameterised};
use utils::cartesian_product;
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

impl Function<[f64], Vec<f64>> for RBFNetwork {
    fn evaluate(&self, inputs: &[f64]) -> Vec<f64> {
        // Compute the feature vector phi:
        let phi = self.phi(inputs);

        // Apply matrix multiplication and return Vec<f64>:
        (self.weights.t().dot(&phi)).into_raw_vec()
    }
}

impl Function<Vec<f64>, Vec<f64>> for RBFNetwork {
    fn evaluate(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.evaluate(inputs.as_slice())
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

impl Parameterised<Vec<f64>, Vec<f64>> for RBFNetwork {
    fn update(&mut self, inputs: &Vec<f64>, errors: &Vec<f64>) {
        self.update(inputs.as_slice(), errors.as_slice())
    }
}


// #[cfg(test)]
// mod tests {
    // use test::Bencher;

    // use {Function, Parameterised};
    // use utils::kernels::Exponential;
    // use geometry::RegularSpace;
    // use geometry::dimensions::Partition;

    // use super::BasisNetwork;

    // #[test]
    // fn test_singular_rbf_network() {
        // let mut ds = RegularSpace::new();
        // ds = ds.push(Partition::new(0.0, 1.0, 1));

        // let bases = vec![Exponential::new(1.0); 1];

        // let mut net = BasisNetwork::new(bases, ds, 1);

        // net.update(&vec![0.5], &vec![5.0]);
        // assert_eq!(net.evaluate(&vec![0.5]), &[5.0]);
    // }

    // #[bench]
    // fn bench_phi(b: &mut Bencher) {
        // let mut ds = RegularSpace::new();
        // ds = ds.push(Partition::new(0.0, 9.0, 10));

        // let bases = vec![Exponential::new(1.0); 10];
        // let mut net = BasisNetwork::new(bases, ds, 1);

        // b.iter(|| net.phi(&vec![1.5]));
    // }

    // #[bench]
    // fn bench_evaluate(b: &mut Bencher) {
        // let mut ds = RegularSpace::new();
        // ds = ds.push(Partition::new(0.0, 9.0, 10));

        // let bases = vec![Exponential::new(1.0); 10];
        // let mut net = BasisNetwork::new(bases, ds, 1);

        // b.iter(|| net.evaluate(&vec![1.5]));
    // }

    // #[bench]
    // fn bench_update(b: &mut Bencher) {
        // let mut ds = RegularSpace::new();
        // ds = ds.push(Partition::new(0.0, 9.0, 10));

        // let bases = vec![Exponential::new(1.0); 10];
        // let mut net = BasisNetwork::new(bases, ds, 1);

        // b.iter(|| net.update(&vec![1.5], &vec![10.0]));
    // }

    // #[bench]
    // fn bench_update_mc(b: &mut Bencher) {
        // let mut ds = RegularSpace::new();
        // ds = ds.push(Partition::new(-1.2, 0.5, 20));
        // ds = ds.push(Partition::new(-0.07, 0.07, 20));

        // let bases = vec![Exponential::new(0.1); 400];
        // let mut net = BasisNetwork::new(bases, ds, 3);

        // b.iter(|| net.update(&vec![0.0, 0.0], &vec![1.0, 0.0, 0.7]));
    // }
// }
