use std::f64::consts::PI;
use super::{Projector, Projection};
use geometry::RegularSpace;
use geometry::norms::l2;
use geometry::dimensions::{BoundedDimension, Continuous};
use ndarray::Array1;
use utils::cartesian_product;


// TODO: Add learning rate scaling http://lis.csail.mit.edu/pubs/konidaris-aaai11a.pdf
// TODO: Add builder which allows use to configure whether to use coefficient scaling or not.


#[derive(Clone, Serialize, Deserialize)]
pub struct Fourier {
    order: u8,
    limits: Vec<(f64, f64)>,
    coefficients: Vec<Vec<f64>>,
}

impl Fourier {
    pub fn new(order: u8, limits: Vec<(f64, f64)>) -> Self {
        let coefficients = Fourier::make_coefficients(order, limits.len());

        Fourier {
            order: order,
            limits: limits,
            coefficients: coefficients,
        }
    }

    pub fn from_space(order: u8, input_space: RegularSpace<Continuous>) -> Self {
        Fourier::new(order, input_space.iter().map(|d| d.limits()).collect())
    }

    fn make_coefficients(order: u8, dim: usize) -> Vec<Vec<f64>> {
        let dcs = vec![(0..(order+1)).map(|v| v as f64).collect::<Vec<f64>>(); dim];
        let mut coefficients = cartesian_product(&dcs).iter().skip(1).map(|cfs| {
            let z = l2(&cfs);

            cfs.iter().map(|c| c/z).collect()
        }).collect::<Vec<Vec<f64>>>();

        coefficients.sort_by(|a, b| b.partial_cmp(a).unwrap());
        coefficients.dedup();

        coefficients
    }
}

impl Projector<RegularSpace<Continuous>> for Fourier {
    fn project(&self, input: &Vec<f64>) -> Projection {
        let scaled_state = input.iter().enumerate().map(|(i, v)| {
            (v - self.limits[i].0) / (self.limits[i].1 - self.limits[i].0)
        }).collect::<Vec<f64>>();

        Projection::Dense(Array1::from_iter(self.coefficients.iter().map(|cfs| {
            let cx = scaled_state.iter().zip(cfs).fold(0.0, |acc, (v, c)| acc + *c*v);

            (PI*cx).cos()
        })))
    }

    fn dim(&self) -> usize {
        self.limits.len()
    }

    fn size(&self) -> usize {
        self.coefficients.len()
    }

    fn activation(&self) -> usize {
        self.size()
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
    fn test_pruning() {
        assert_eq!(Fourier::new(1, vec![(0.0, 1.0)]).dim(), 1);
        assert_eq!(Fourier::new(2, vec![(0.0, 1.0)]).dim(), 1);
        assert_eq!(Fourier::new(3, vec![(0.0, 1.0)]).dim(), 1);
        assert_eq!(Fourier::new(4, vec![(0.0, 1.0)]).dim(), 1);
        assert_eq!(Fourier::new(5, vec![(0.0, 1.0)]).dim(), 1);
    }

    #[test]
    fn test_symmetry() {
        let f = Fourier::new(1, vec![(0.0, 1.0)]);

        assert_eq!(f.project(&vec![-1.0]), f.project(&vec![1.0]));
        assert_eq!(f.project(&vec![-0.5]), f.project(&vec![0.5]));
    }

    #[test]
    fn test_order1_1d() {
        let f = Fourier::new(1, vec![(0.0, 1.0)]);

        assert_eq!(f.dim(), 1);
        assert_eq!(f.size(), 1);

        assert!(f.project(&vec![-1.0]).all_close(&arr1(&vec![-1.0]), 1e-6));
        assert!(f.project(&vec![-0.5]).all_close(&arr1(&vec![0.0]), 1e-6));
        assert!(f.project(&vec![0.0]).all_close(&arr1(&vec![1.0]), 1e-6));
        assert!(f.project(&vec![0.5]).all_close(&arr1(&vec![0.0]), 1e-6));
        assert!(f.project(&vec![1.0]).all_close(&arr1(&vec![-1.0]), 1e-6));

        assert!(f.project(&vec![-2.0/3.0]).all_close(&arr1(&vec![-1.0]), 1e-6));
        assert!(f.project(&vec![-1.0/3.0]).all_close(&arr1(&vec![1.0]), 1e-6));
        assert!(f.project(&vec![1.0/3.0]).all_close(&arr1(&vec![1.0]), 1e-6));
        assert!(f.project(&vec![2.0/3.0]).all_close(&arr1(&vec![-1.0]), 1e-6));
    }

    #[test]
    fn test_order2_1d() {
        let f1 = Fourier::new(1, vec![(0.0, 1.0)]);
        let f2 = Fourier::new(2, vec![(0.0, 1.0)]);

        assert_eq!(f2.dim(), f1.dim());
        assert_eq!(f2.size(), f2.size());

        assert_eq!(f2.project(&vec![-1.0]), f1.project(&vec![-1.0]));
        assert_eq!(f2.project(&vec![-0.5]), f1.project(&vec![-0.5]));
        assert_eq!(f2.project(&vec![0.0]), f1.project(&vec![0.0]));
        assert_eq!(f2.project(&vec![0.5]), f1.project(&vec![0.5]));
        assert_eq!(f2.project(&vec![1.0]), f1.project(&vec![1.0]));


        assert_eq!(f2.project(&vec![-2.0/3.0]), f1.project(&vec![-2.0/3.0]));
        assert_eq!(f2.project(&vec![-1.0/3.0]), f1.project(&vec![-1.0/3.0]));
        assert_eq!(f2.project(&vec![1.0/3.0]), f1.project(&vec![1.0/3.0]));
        assert_eq!(f2.project(&vec![2.0/3.0]), f1.project(&vec![2.0/3.0]));
    }

    #[test]
    fn test_order1_2d() {
        let f = Fourier::new(1, vec![(0.0, 1.0), (5.0, 6.0)]);

        assert_eq!(f.dim(), 2);
        assert_eq!(f.size(), 3);

        assert!(f.project(&vec![0.0, 5.0]).all_close(&arr1(&vec![1.0/3.0; 3]), 1e-6));
        assert!(f.project(&vec![0.5, 5.0]).all_close(&arr1(&vec![4.24042024e-17, 3.07486821e-1, 6.92513179e-1]), 1e-6));
        assert!(f.project(&vec![0.0, 5.5]).all_close(&arr1(&vec![6.92513179e-1, 3.07486821e-1, 4.24042024e-17]), 1e-6));
        assert!(f.project(&vec![0.5, 5.5]).all_close(&arr1(&vec![1.01093534e-16, -1.0, 1.01093534e-16]), 1e-6));
        assert!(f.project(&vec![1.0, 5.5]).all_close(&arr1(&vec![-5.04567213e-1, -4.95432787e-1, 3.08958311e-17]), 1e-6));
        assert!(f.project(&vec![0.5, 6.0]).all_close(&arr1(&vec![3.08958311e-17, -4.95432787e-1, -5.04567213e-1]), 1e-6));
        assert!(f.project(&vec![1.0, 6.0]).all_close(&arr1(&vec![-0.44125654, -0.11748691, -0.44125654]), 1e-6));
    }

    #[test]
    fn test_order2_2d() {
        let f = Fourier::new(2, vec![(0.0, 1.0), (5.0, 6.0)]);

        assert_eq!(f.dim(), 2);
        assert_eq!(f.size(), 5);

        assert!(f.project(&vec![0.0, 5.0]).all_close(&arr1(&vec![0.2; 5]), 1e-6));
        assert!(f.project(&vec![0.5, 5.0]).all_close(&arr1(&vec![2.58110397e-17, 6.95831686e-2,
                                                                 1.87164340e-1, 3.21726225e-1,
                                                                 4.21526267e-1]), 1e-6));
        assert!(f.project(&vec![0.5, 5.5]).all_close(&arr1(&vec![3.76070090e-17, -3.13998939e-1,
                                                                 -3.72002121e-1, -3.13998939e-1,
                                                                 3.76070090e-17]), 1e-6));
        assert!(f.project(&vec![0.5, 6.0]).all_close(&arr1(&vec![1.58656439e-17, -2.44984612e-1,
                                                                 -2.54414913e-1, -2.41494847e-1,
                                                                 -2.59105628e-1]), 1e-6));
        assert!(f.project(&vec![1.0, 6.0]).all_close(&arr1(&vec![-0.31048999, -0.1481752,
                                                                 -0.08266962, -0.1481752,
                                                                 -0.31048999]), 1e-6));
    }
}
