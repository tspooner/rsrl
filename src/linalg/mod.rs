use ndarray::{ArrayBase, Array2, DataMut, Ix2};

pub struct Entry {
    pub index: [usize; 2],
    pub gradient: f64,
}

pub trait MatrixLike: Clone + Into<Array2<f64>> {
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

    fn addto<D: DataMut<Elem = f64>>(&self, arr: &mut ArrayBase<D, Ix2>) {
        self.for_each(|pd| arr[pd.index] += pd.gradient);
    }

    fn scaled_addto<D: DataMut<Elem = f64>>(&self, alpha: f64, arr: &mut ArrayBase<D, Ix2>) {
        self.for_each(|pd| arr[pd.index] += alpha * pd.gradient);
    }

    fn to_dense(&self) -> Array2<f64> { self.clone().into() }
}

import_all!(dense);
import_all!(columnar);
import_all!(sparse);

// import_all!(arithmetic);
// import_all!(scaling);
// import_all!(shifting);
