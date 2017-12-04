use super::Projection;
use geometry::{Span, Space, RegularSpace};
use geometry::dimensions::{Continuous, Partitioned};
use ndarray::{Axis, ArrayView, Array1, Array2};
use utils::cartesian_product;


#[derive(Serialize, Deserialize)]
pub struct RBFNetwork {
    mu: Array2<f64>,
    beta: Array1<f64>,
}

impl RBFNetwork {
    pub fn new(mu: Array2<f64>, sigma: Array1<f64>) -> Self {
        if mu.cols() != sigma.len() {
            panic!("Dimensions of mu ({:?}) and sigma ({:?}) must agree.",
                   mu.shape(),
                   sigma.shape());
        }

        RBFNetwork {
            mu: mu,
            beta: 0.5 / sigma.map(|v| v*v),
        }
    }

    pub fn from_space(input_space: RegularSpace<Partitioned>) -> Self {
        let n_features = match input_space.span() {
            Span::Finite(s) => s,
            _ => { panic!("`RBFNetwork` projection only supports partitioned input spaces.") }
        };

        let centres = input_space.centres();
        let flat_combs = cartesian_product(&centres)
            .iter()
            .cloned()
            .flat_map(|e| e)
            .collect();

        let mu = Array2::from_shape_vec((n_features, input_space.dim()), flat_combs).unwrap();
        let sigma = input_space.iter().map(|d| d.partition_width()).collect();

        RBFNetwork::new(mu, sigma)
    }

    pub fn kernel(&self, input: &[f64]) -> Array1<f64> {
        let d = &self.mu - &ArrayView::from_shape((1, self.mu.cols()), input).unwrap();

        (&d*&d*&self.beta).mapv(|v| (-v.abs()).exp()).sum_axis(Axis(1))
    }
}

impl Projection<RegularSpace<Continuous>> for RBFNetwork {
    fn project_onto(&self, input: &Vec<f64>, phi: &mut Array1<f64>) {
        let p = self.kernel(input);

        phi.scaled_add(1.0/p.scalar_sum(), &p)
    }

    fn dim(&self) -> usize {
        self.mu.cols()
    }

    fn size(&self) -> usize {
        self.mu.rows()
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.mu == other.mu && self.beta == other.beta && self.size() == other.size()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_size() {
        assert_eq!(RBFNetwork::new(arr2(&[[0.0]]), arr1(&[0.25])).size(), 1);
        assert_eq!(RBFNetwork::new(arr2(&[[0.0], [0.5], [1.0]]), arr1(&[0.25])).size(), 3);
        assert_eq!(RBFNetwork::new(arr2(&[[0.0]; 10]), arr1(&[0.25])).size(), 10);
        assert_eq!(RBFNetwork::new(arr2(&[[0.0]; 100]), arr1(&[0.25])).size(), 100);
    }

    #[test]
    fn test_dimensionality() {
        assert_eq!(RBFNetwork::new(arr2(&[[0.0], [0.5], [1.0]]), arr1(&[0.25])).dim(), 1);
        assert_eq!(RBFNetwork::new(arr2(&[[0.0, 0.5, 1.0]; 10]), arr1(&[0.1, 0.2, 0.3])).dim(), 3);
    }

    #[test]
    fn test_kernel_relevance() {
        let rbf = RBFNetwork::new(arr2(&[[0.0]]), arr1(&[0.25]));
        let mut phi = rbf.kernel(&vec![0.0]);

        for i in 1..10 {
            let phi_new = rbf.kernel(&vec![i as f64 / 10.0]);
            assert!(phi_new[0] < phi[0]);

            phi = phi_new
        }
    }

    #[test]
    fn test_kernel_isotropy() {
        let rbf = RBFNetwork::new(arr2(&[[0.0]]), arr1(&[0.25]));
        let phi = rbf.kernel(&vec![0.0]);

        for i in 1..10 {
            let phi_left = rbf.kernel(&vec![-i as f64 / 10.0]);
            let phi_right = rbf.kernel(&vec![i as f64 / 10.0]);

            assert!(phi_left[0] < phi[0]);
            assert!(phi_right[0] < phi[0]);
            assert_eq!(phi_left, phi_right);
        }
    }

    #[test]
    fn test_projection_1d() {
        let rbf = RBFNetwork::new(arr2(&[[0.0], [0.5], [1.0]]), arr1(&[0.25]));
        let phi = rbf.project(&vec![0.25]);

        assert!(phi.all_close(&arr1(&[0.49546264, 0.49546264, 0.00907471]), 1e-6));
        assert_eq!(phi.scalar_sum(), 1.0);
    }

    #[test]
    fn test_projection_2d() {
        let rbf = RBFNetwork::new(arr2(&[[0.0, -10.0], [0.5, -8.0], [1.0, -6.0]]),
                                  arr1(&[0.25, 2.0]));
        let phi = rbf.project(&vec![0.67, -7.0]);

        assert!(phi.all_close(&arr1(&[0.10579518, 0.50344131, 0.3907635]), 1e-6));
        assert_eq!(phi.scalar_sum(), 1.0);
    }
}
