use std::hash::{Hasher, BuildHasher};
use super::{Projection, SparseProjection};
use geometry::RegularSpace;
use geometry::dimensions::Continuous;
use ndarray::Array1;


#[derive(Serialize, Deserialize)]
pub struct TileCoding<H: BuildHasher> {
    hasher_builder: H,
    n_tilings: usize,
    memory_size: usize,
}

impl<H: BuildHasher> TileCoding<H> {
    pub fn new(hasher_builder: H, n_tilings: usize, memory_size: usize) -> Self {
        TileCoding {
            hasher_builder: hasher_builder,
            n_tilings: n_tilings,
            memory_size: memory_size,
        }
    }
}

impl<H: BuildHasher> Projection<RegularSpace<Continuous>> for TileCoding<H> {
    fn project_onto(&self, input: &Vec<f64>, phi: &mut Array1<f64>) {
        for t in self.project_sparse(input).iter() {
            phi[*t] = 1.0;
        }
    }

    fn dim(&self) -> usize {
        self.memory_size as usize
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.dim() == other.dim() && self.n_tilings == other.n_tilings &&
        self.memory_size == other.memory_size
    }
}

impl<H: BuildHasher> SparseProjection<RegularSpace<Continuous>> for TileCoding<H> {
    fn project_onto_sparse(&self, input: &Vec<f64>, indices: &mut Array1<usize>) {
        let mut hasher = self.hasher_builder.build_hasher();

        let n_floats = input.len();
        let floats: Vec<usize> =
            input.iter().map(|f| (*f * self.n_tilings as f64).floor() as usize).collect();

        for t in 0..self.n_tilings {
            hasher.write_usize(t);
            for i in 0..n_floats {
                hasher.write_usize((floats[i] + t + i*t*2) / self.n_tilings)
            }

            indices[t] = hasher.finish() as usize % self.memory_size;
        }
    }

    fn dim_activation(&self) -> usize {
        self.n_tilings
    }
}
