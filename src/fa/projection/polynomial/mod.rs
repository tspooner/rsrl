use super::{Projector, Projection};
use geometry::RegularSpace;
use geometry::dimensions::{BoundedDimension, Continuous};
use ndarray::Array1;
use utils::cartesian_product;


mod cpfk;


/// Polynomial basis projector.
#[derive(Clone, Serialize, Deserialize)]
pub struct Polynomial {
    order: u8,
    limits: Vec<(f64, f64)>,
    exponents: Vec<Vec<i32>>,
}

impl Polynomial {
    pub fn new(order: u8, limits: Vec<(f64, f64)>) -> Self {
        let exponents = Polynomial::make_exponents(order, limits.len());

        Polynomial {
            order: order,
            limits: limits,
            exponents: exponents,
        }
    }

    pub fn from_space(order: u8, input_space: RegularSpace<Continuous>) -> Self {
        Polynomial::new(order, input_space.iter().map(|d| d.limits()).collect())
    }

    fn make_exponents(order: u8, dim: usize) -> Vec<Vec<i32>> {
        let dcs = vec![(0..(order+1)).map(|v| v as i32).collect::<Vec<i32>>(); dim];
        let mut exponents = cartesian_product(&dcs);

        exponents.sort_by(|a, b| b.partial_cmp(a).unwrap());
        exponents.dedup();

        exponents
    }
}

impl Projector<RegularSpace<Continuous>> for Polynomial {
    fn project(&self, input: &Vec<f64>) -> Projection {
        let scaled_state = input.iter().enumerate().map(|(i, v)| {
            (v - self.limits[i].0) / (self.limits[i].1 - self.limits[i].0)
        }).map(|v| 2.0*v - 1.0).collect::<Vec<f64>>();

        let activations = self.exponents.iter().map(|exps| {
            scaled_state.iter().zip(exps).map(|(v, e)| v.powi(*e)).product()
        });

        Projection::Dense(Array1::from_iter(activations))
    }

    fn dim(&self) -> usize {
        self.limits.len()
    }

    fn size(&self) -> usize {
        self.exponents.len()
    }

    fn activation(&self) -> usize {
        self.size()
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.order == other.order && self.limits == other.limits
    }
}
