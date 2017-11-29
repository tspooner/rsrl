use super::Projection;
use geometry::{Span, Space, RegularSpace};
use geometry::dimensions::{Continuous, Partitioned};
use ndarray::{Axis, ArrayView, Array1, Array2};
use utils::cartesian_product;


#[derive(Serialize, Deserialize)]
pub struct RBFNetwork {
    mu: Array2<f64>,
    gamma: Array1<f64>,
}

impl RBFNetwork {
    pub fn new(mu: Array2<f64>, gamma: Array1<f64>) -> Self {
        if mu.cols() != gamma.len() {
            panic!("Dimensions of mu ({:?}) and gamma ({:?}) must agree.",
                   mu.shape(),
                   gamma.shape());
        }

        RBFNetwork {
            mu: mu,
            gamma: gamma,
        }
    }

    pub fn from_space(input_space: RegularSpace<Partitioned>) -> Self {
        let n_features = match input_space.span() {
            Span::Finite(s) => s,
            _ => {
                panic!("`RBFNetwork` projection only supports partitioned \
                        input spaces.")
            }
        };

        let centres = input_space.centres();
        let flat_combs = cartesian_product(&centres)
            .iter()
            .cloned()
            .flat_map(|e| e)
            .collect();

        let mu = Array2::from_shape_vec((n_features, input_space.dim()), flat_combs).unwrap();
        let gamma = input_space.iter()
            .map(|d| {
                let s = d.partition_width();
                -1.0 / (s * s)
            })
            .collect();

        RBFNetwork::new(mu, gamma)
    }
}

impl Projection<RegularSpace<Continuous>> for RBFNetwork {
    fn project_onto(&self, input: &Vec<f64>, phi: &mut Array1<f64>) {
        let d = &self.mu - &ArrayView::from_shape((1, self.mu.cols()), input).unwrap();
        let p = (&d*&d*&self.gamma).mapv(|v| v.exp()).sum_axis(Axis(1));

        phi.scaled_add(1.0/p.scalar_sum(), &p)
    }

    fn dim(&self) -> usize {
        self.mu.rows()
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.mu == other.mu && self.gamma == other.gamma && self.dim() == other.dim()
    }
}
