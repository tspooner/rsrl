use super::{Function, Parameterised, Linear, VFunction, QFunction};

use utils::dot;
use ndarray::{arr1, Array1, Array2};
use geometry::{Space, RegularSpace};
use geometry::dimensions::{Dimension, Partitioned, Continuous};


/// Linearly partitioned function representation.
pub struct UniformGrid {
    weights: Array2<f64>,
    input_space: RegularSpace<Partitioned>,
}

impl UniformGrid {
    pub fn new(input_space: RegularSpace<Partitioned>,
               n_outputs: usize) -> Self
    {
        let n_features = input_space.span().into();

        UniformGrid {
            weights: Array2::<f64>::zeros((n_features, n_outputs)),
            input_space: input_space,
        }
    }

    fn hash(&self, input: &[f64]) -> usize {
        let mut in_it = input.iter().rev();
        let mut d_it = self.input_space.iter().rev();

        let acc = d_it.next().unwrap().convert(*in_it.next().unwrap());

        d_it.zip(in_it)
            .fold(acc, |acc, (d, v)| d.convert(*v) + d.density() * acc)
    }
}


impl Function<Vec<f64>, f64> for UniformGrid {
    fn evaluate(&self, input: &Vec<f64>) -> f64 {
        // Hash the input down to an index:
        self.weights[[self.hash(input), 0]]
    }
}

impl Function<Vec<f64>, Vec<f64>> for UniformGrid {
    fn evaluate(&self, input: &Vec<f64>) -> Vec<f64> {
        // Hash the input down to a row index:
        let ri = self.hash(input);

        // Get the row slice and convert to a Vec<f64>:
        self.weights.row(ri).to_vec()
    }
}


impl Parameterised<Vec<f64>, f64> for UniformGrid {
    fn update(&mut self, input: &Vec<f64>, error: f64) {
        let index = self.hash(input);

        unsafe {
            *self.weights.uget_mut((index, 0)) += error
        }
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.weights.shape() == other.weights.shape()
    }
}

impl Parameterised<Vec<f64>, Vec<f64>> for UniformGrid {
    fn update(&mut self, input: &Vec<f64>, errors: Vec<f64>) {
        // Hash the input down to a row index:
        let ri = self.hash(input);

        // Get the row slice and perform update via memcpy:
        self.weights.row_mut(ri).scaled_add(1.0, &arr1(&errors));
    }

    fn equivalent(&self, other: &Self) -> bool {
        <Self as Parameterised<Vec<f64>, f64>>::equivalent(self, other)
    }
}


impl Linear<RegularSpace<Continuous>> for UniformGrid
{
    fn phi(&self, input: &Vec<f64>) -> Array1<f64> {
        let mut p = Array1::<f64>::zeros(self.weights.rows());
        p[self.hash(input)] = 1.0;

        p
    }
}


impl VFunction<RegularSpace<Continuous>> for UniformGrid
{
    fn evaluate_phi(&self, phi: &Array1<f64>) -> f64 {
        dot(self.weights.column(0).as_slice().unwrap(),
            phi.as_slice().unwrap())
    }

    fn update_phi(&mut self, phi: &Array1<f64>, error: f64) {
        self.weights.column_mut(0).scaled_add(error, phi);
    }
}


impl QFunction<RegularSpace<Continuous>> for UniformGrid
{
    fn evaluate_action(&self, input: &Vec<f64>, action: usize) -> f64 {
        // Hash the input down to a row index:
        let ri = self.hash(input);

        self.weights[[ri, action]]
    }

    fn update_action(&mut self, input: &Vec<f64>, action: usize, error: f64) {
        let index = self.hash(input);

        unsafe {
            *self.weights.uget_mut((index, action)) += error
        }
    }
}


#[cfg(test)]
mod tests {
    use super::UniformGrid;

    use fa::{Function, Parameterised, VFunction, QFunction};
    use geometry::RegularSpace;
    use geometry::dimensions::Partitioned;

    #[test]
    fn test_update_eval() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));

        let mut t = UniformGrid::new(ds, 1);

        t.update(&vec![1.5], 25.5);

        let out: f64 = t.evaluate(&vec![1.5]);
        assert_eq!(out, 25.5);

        t.update(&vec![1.5], -12.75);

        let out: f64 = t.evaluate(&vec![1.5]);
        assert_eq!(out, 12.75);
    }

    #[test]
    fn test_multiple_outputs() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));

        let mut t = UniformGrid::new(ds, 2);

        t.update(&vec![1.5], vec![-20.1, 25.5]);

        let out: Vec<f64> = t.evaluate(&vec![1.5]);
        assert_eq!(out, vec![-20.1, 25.5]);

        t.update_action(&vec![1.5], 0, 20.1);

        let out: f64 = t.evaluate_action(&vec![1.5], 0);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_generalisation() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));

        let mut t = UniformGrid::new(ds, 1);

        t.update(&vec![0.5], vec![1.2]);

        for i in 1..10 {
            let out: f64 = t.evaluate(&vec![i as f64 / 10.0]);
            assert_eq!(out, 1.2);
        }
    }

    #[test]
    fn test_1d() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));

        let mut t = UniformGrid::new(ds, 1);

        for i in 0..10 {
            let input: Vec<f64> = vec![i as u32 as f64];

            let out: f64 = t.evaluate(&input);
            assert_eq!(out, 0.0);

            t.update(&input, vec![1.0]);

            let out: f64 = t.evaluate(&input);
            assert_eq!(out, 1.0);
        }
    }

    #[test]
    fn test_2d() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));

        let mut t = UniformGrid::new(ds, 1);

        for i in 0..10 {
            for j in 0..10 {
                let input: Vec<f64> = vec![i as u32 as f64, j as u32 as f64];

                let out: f64 = t.evaluate(&input);
                assert_eq!(out, 0.0);

                t.update(&input, vec![1.0]);

                let out: f64 = t.evaluate(&input);
                assert_eq!(out, 1.0);
            }
        }
    }

    #[test]
    fn test_3d() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));
        ds = ds.push(Partitioned::new(0.0, 10.0, 10));

        let mut t = UniformGrid::new(ds, 1);

        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let input: Vec<f64> = vec![i as u32 as f64, j as u32 as f64, k as u32 as f64];

                    let out: f64 = t.evaluate(&input);
                    assert_eq!(out, 0.0);

                    t.update(&input, vec![1.0]);

                    let out: f64 = t.evaluate(&input);
                    assert_eq!(out, 1.0);
                }
            }
        }
    }
}
