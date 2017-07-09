use super::Projection;
use ndarray::Array1;
use geometry::{Space, RegularSpace};
use geometry::dimensions::{Dimension, Partitioned};


pub struct UniformGrid {
    n_features: usize,
    input_space: RegularSpace<Partitioned>,
}

impl UniformGrid {
    pub fn new(input_space: RegularSpace<Partitioned>) -> Self {
        let n_features = input_space.span().into();

        UniformGrid {
            n_features: n_features,
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

impl Projection for UniformGrid {
    fn project(&self, input: &[f64]) -> Array1<f64> {
        let mut p = Array1::<f64>::zeros(self.n_features);
        p[self.hash(input)] = 1.0;

        p
    }

    fn dim(&self) -> usize {
        self.n_features
    }
}
