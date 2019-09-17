use crate::geometry::{Matrix, MatrixViewMut};

pub struct Entry {
    pub index: [usize; 2],
    pub gradient: f64,
}

pub trait MatrixLike: Clone + Into<Matrix<f64>> {
    fn zeros(dim: [usize; 2]) -> Self;

    fn dim(&self) -> [usize; 2];

    fn n_rows(&self) -> usize { self.dim()[0] }

    fn n_cols(&self) -> usize { self.dim()[1] }

    fn map(mut self, f: impl Fn(f64) -> f64) -> Self {
        self.map_inplace(f); self
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64);

    fn combine(mut self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        self.combine_inplace(other, f); self
    }

    fn combine_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64);

    fn for_each(&self, f: impl FnMut(Entry));

    fn addto(&self, weights: &mut MatrixViewMut<f64>) {
        self.for_each(|pd| weights[pd.index] += pd.gradient);
    }

    fn scaled_addto(&self, alpha: f64, weights: &mut MatrixViewMut<f64>) {
        self.for_each(|pd| weights[pd.index] += alpha * pd.gradient);
    }

    fn to_dense(&self) -> Matrix<f64> { self.clone().into() }
}

import_all!(dense);
import_all!(columnar);
import_all!(sparse);

// import_all!(arithmetic);
// import_all!(scaling);
// import_all!(shifting);
