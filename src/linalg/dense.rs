use ndarray::Array2;
use std::ops::AddAssign;
use super::*;

impl MatrixLike for Array2<f64> {
    fn zeros(dim: [usize; 2]) -> Self { Self::zeros(dim) }

    fn dim(&self) -> [usize; 2] { let (r, c) = self.dim(); [r, c] }

    fn map(self, f: impl Fn(f64) -> f64) -> Self { self.mapv_into(f) }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) { self.mapv_inplace(f); }

    fn combine_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        self.zip_mut_with(other, |x, y| *x = f(*x, *y));
    }

    fn for_each(&self, f: impl FnMut(Entry)) {
        self.indexed_iter().map(|(index, gradient)| Entry {
            index: [index.0, index.1],
            gradient: *gradient,
        }).for_each(f);
    }

    fn addto<D: DataMut<Elem = f64>>(&self, weights: &mut ArrayBase<D, Ix2>) {
        weights.add_assign(self);
    }

    fn scaled_addto<D: DataMut<Elem = f64>>(&self, alpha: f64, weights: &mut ArrayBase<D, Ix2>) {
        weights.scaled_add(alpha, self);
    }
}
