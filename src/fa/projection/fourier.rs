use std::f64::consts::PI;
use super::Projection;
use geometry::RegularSpace;
use geometry::dimensions::{BoundedDimension, Continuous};
use ndarray::Array1;
use utils::cartesian_product;


#[derive(Serialize, Deserialize)]
pub struct Fourier {
    order: u8,
    limits: Vec<(f64, f64)>,
    coefficients: Vec<Vec<u8>>,
}

impl Fourier {
    pub fn new(order: u8, limits: Vec<(f64, f64)>) -> Self {
        let coefficients =
            cartesian_product(&vec![(0..(order+1)).collect::<Vec<u8>>(); limits.len()]);

        Fourier {
            order: order,
            limits: limits,
            coefficients: coefficients,
        }
    }

    pub fn from_space(order: u8, input_space: RegularSpace<Continuous>) -> Self {
        Fourier::new(order, input_space.iter().map(|d| d.limits()).collect())
    }
}

impl Projection<RegularSpace<Continuous>> for Fourier {
    fn project_onto(&self, input: &Vec<f64>, phi: &mut Array1<f64>) {
        let scaled_state = input.iter().enumerate().map(|(i, v)| {
            (v - self.limits[i].0) / (self.limits[i].1 - self.limits[i].0)
        }).collect::<Vec<f64>>();

        for (i, cfs) in self.coefficients.iter().enumerate() {
            let cx = scaled_state.iter().zip(cfs).fold(0.0, |acc, (v, c)| acc + (*c as f64)*v);

            phi[i] = (PI*cx).cos();
        }
    }

    fn dim(&self) -> usize {
        self.limits.len()
    }

    fn size(&self) -> usize {
        (self.order as usize + 1).pow(self.dim() as u32)
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.order == other.order && self.limits == other.limits
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_order1_1d() {
        let f = Fourier::new(1, vec![(0.0, 1.0)]);

        assert_eq!(f.dim(), 1);
        assert_eq!(f.size(), 2);

        assert!(f.project(&vec![-1.0]).all_close(&arr1(&vec![1.0, -1.0]), 1e-6));
        assert!(f.project(&vec![-0.5]).all_close(&arr1(&vec![1.0, 0.0]), 1e-6));
        assert!(f.project(&vec![0.0]).all_close(&arr1(&vec![1.0, 1.0]), 1e-6));
        assert!(f.project(&vec![0.5]).all_close(&arr1(&vec![1.0, 0.0]), 1e-6));
        assert!(f.project(&vec![1.0]).all_close(&arr1(&vec![1.0, -1.0]), 1e-6));


        assert!(f.project(&vec![-2.0/3.0]).all_close(&arr1(&vec![1.0, -0.5]), 1e-6));
        assert!(f.project(&vec![-1.0/3.0]).all_close(&arr1(&vec![1.0, 0.5]), 1e-6));
        assert!(f.project(&vec![1.0/3.0]).all_close(&arr1(&vec![1.0, 0.5]), 1e-6));
        assert!(f.project(&vec![2.0/3.0]).all_close(&arr1(&vec![1.0, -0.5]), 1e-6));
    }
}
