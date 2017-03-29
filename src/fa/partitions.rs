use {Function, Parameterised};
use ndarray::{arr1, Array2};
use geometry::{Span, Space, RegularSpace};
use geometry::dimensions::Partition;

use std::ops::AddAssign;


pub struct Partitions {
    weights: Array2<f64>,
    input_space: RegularSpace<Partition>,
}

impl Partitions {
    pub fn new(input_space: RegularSpace<Partition>, n_outputs: usize) -> Self {
        let n_features = match input_space.span() {
            Span::Finite(s) => s,
            _ => panic!("`Partitions` function approximator only supports \
                         finite input spaces.")
        };

        Partitions {
            weights: Array2::<f64>::zeros((n_features, n_outputs)),
            input_space: input_space,
        }
    }

    fn hash(&self, inputs: &[f64]) -> usize {
        let mut in_it = inputs.iter().rev();
        let mut d_it = self.input_space.iter().rev();

        let acc = d_it.next().unwrap().to_partition(in_it.next().unwrap());

        in_it.zip(d_it).fold(acc, |acc, (v, d)| {
            let i = d.to_partition(v);

            i + d.density() * acc
        })
    }
}

impl Function<[f64], Vec<f64>> for Partitions {
    fn evaluate(&self, inputs: &[f64]) -> Vec<f64> {
        // Hash the inputs down to a row index:
        let ri = self.hash(inputs);

        // Get the row slice and convert to a Vec<f64>:
        self.weights.row(ri).to_vec()
    }
}

impl Function<Vec<f64>, Vec<f64>> for Partitions {
    fn evaluate(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.evaluate(inputs.as_slice())
    }
}

impl Parameterised<[f64], Vec<f64>> for Partitions {
    fn update(&mut self, inputs: &[f64], errors: &Vec<f64>) {
        // Hash the inputs down to a row index:
        let ri = self.hash(inputs);

        // Get the row slice and perform update via memcpy:
        self.weights.row_mut(ri).add_assign(&arr1(errors));
    }
}

impl Parameterised<Vec<f64>, Vec<f64>> for Partitions {
    fn update(&mut self, inputs: &Vec<f64>, errors: &Vec<f64>) {
        self.update(inputs.as_slice(), errors)
    }
}


#[cfg(test)]
mod tests {
    use super::Partitions;

    use {Function, Parameterised};
    use geometry::RegularSpace;
    use geometry::dimensions::Partition;

    #[test]
    fn test_update_eval() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partition::new(0.0, 9.0, 10));

        let mut t = Partitions::new(ds, 1);

        t.update(&vec![1.5], &vec![25.5]);
        assert_eq!(t.evaluate(&vec![1.5]), &[25.5]);

        t.update(&vec![1.5], &vec![-12.75]);
        assert_eq!(t.evaluate(&vec![1.5]), &[12.75]);
    }

    #[test]
    fn test_collisions() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partition::new(0.0, 9.0, 10));

        let mut t = Partitions::new(ds, 1);

        t.update(&vec![0.5], &vec![1.2]);
        assert_eq!(t.evaluate(&vec![0.2]), &[1.2]);
        assert_eq!(t.evaluate(&vec![0.5]), &[1.2]);
        assert_eq!(t.evaluate(&vec![0.8]), &[1.2]);
    }

    #[test]
    fn test_1d() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partition::new(0.0, 9.0, 10));

        let mut t = Partitions::new(ds, 1);

        for i in 0..10 {
            let inputs: Vec<f64> = vec![i as u32 as f64];

            assert_eq!(t.evaluate(&inputs), &[0.0]);

            t.update(&inputs, &vec![1.0]);
            assert_eq!(t.evaluate(&inputs), &[1.0]);
        }
    }

    #[test]
    fn test_2d() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partition::new(0.0, 9.0, 10));
        ds = ds.push(Partition::new(0.0, 9.0, 10));

        let mut t = Partitions::new(ds, 1);

        for i in 0..10 {
            for j in 0..10 {
                let inputs: Vec<f64> = vec![i as u32 as f64, j as u32 as f64];

                assert_eq!(t.evaluate(&inputs), &[0.0]);

                t.update(&inputs, &vec![1.0]);
                assert_eq!(t.evaluate(&inputs), &[1.0]);
            }
        }
    }

    #[test]
    fn test_3d() {
        let mut ds = RegularSpace::new();
        ds = ds.push(Partition::new(0.0, 9.0, 10));
        ds = ds.push(Partition::new(0.0, 9.0, 10));
        ds = ds.push(Partition::new(0.0, 9.0, 10));

        let mut t = Partitions::new(ds, 1);

        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let inputs: Vec<f64> = vec![i as u32 as f64, j as u32 as f64, k as u32 as f64];

                    assert_eq!(t.evaluate(&inputs), &[0.0]);

                    t.update(&inputs, &vec![1.0]);
                    assert_eq!(t.evaluate(&inputs), &[1.0]);
                }
            }
        }
    }
}
