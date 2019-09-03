use crate::{geometry::{Matrix, MatrixView, MatrixViewMut}};
use std::ops::{Add, AddAssign, Mul, MulAssign};
use super::{Gradient, PartialDerivative};

impl Gradient for Matrix<f64> {
    fn dim(&self) -> [usize; 2] { let (r, c) = self.dim(); [r, c] }

    fn map(self, f: impl Fn(f64) -> f64) -> Self { self.mapv_into(f) }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) { self.mapv_inplace(f); }

    fn combine_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        self.zip_mut_with(other, |x, y| *x = f(*x, *y));
    }

    fn for_each(&self, f: impl FnMut(PartialDerivative)) {
        self.indexed_iter().map(|(index, gradient)| PartialDerivative {
            index: [index.0, index.1],
            gradient: *gradient,
        }).for_each(f);
    }

    fn addto(&self, weights: &mut MatrixViewMut<f64>) {
        weights.add_assign(self);
    }

    fn scaled_addto(&self, alpha: f64, weights: &mut MatrixViewMut<f64>) {
        weights.scaled_add(alpha, self);
    }
}
