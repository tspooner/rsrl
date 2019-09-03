use geometry::{Vector, Matrix, MatrixViewMut};
use std::ops::{Add, AddAssign, Mul, MulAssign};
use super::{Gradient, PartialDerivative};

#[derive(Clone, Debug, Default)]
pub struct Column {
    dim: [usize; 2],
    grad: Vector<f64>,
}

impl Column {
    pub fn new(n_cols: usize, grad: Vector<f64>) -> Column {
        Column { dim: [grad.len(), n_cols], grad }
    }

    pub fn empty(dim: [usize; 2]) -> Self {
        Column { dim, grads: Vector::zeros(dim[0]) }
    }
}

impl Gradient for Column {
    fn dim(&self) -> [usize; 2] { self.dim }

    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Column {
            dim: self.dim,
            grads: self.grads.into_iter().map(|(c, pds)| (c, pds.mapv_into(&f))).collect(),
        }
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.grads.values_mut().for_each(|pds| pds.mapv_inplace(&f));
    }

    fn combine_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        for (c, xs) in self.grads.iter_mut() {
            if let Some(ys) = other.grads.get(c) {
                xs.zip_mut_with(ys, |x, y| *x = f(*x, *y));
            } else {
                xs.mapv_inplace(|x| f(x, 0.0));
            }
        }

        for (&c, ys) in other.grads.iter() {
            self.grads.entry(c).or_insert_with(|| ys.mapv(|y| f(0.0, y)));
        }
    }

    fn for_each(&self, mut f: impl FnMut(PartialDerivative)) {
        for (&c, pds) in self.grads.iter() {
            pds.indexed_iter().map(|(row, gradient)| PartialDerivative {
                index: [row, c],
                gradient: *gradient,
            }).for_each(|pd| f(pd));
        }
    }

    fn addto(&self, weights: &mut MatrixViewMut<f64>) {
        for (&c, pds) in self.grads.iter() {
            weights.column_mut(c).add_assign(pds);
        }
    }

    fn scaled_addto(&self, alpha: f64, weights: &mut MatrixViewMut<f64>) {
        for (&c, pds) in self.grads.iter() {
            weights.column_mut(c).scaled_add(alpha, pds);
        }
    }
}

impl Into<Matrix<f64>> for Column {
    fn into(self) -> Matrix<f64> {
        let mut g_matrix = Matrix::zeros((self.dim[0], self.dim[1]));

        for (c, g_vector) in self.grads.into_iter() {
            g_matrix.column_mut(c).assign(&g_vector);
        }

        g_matrix
    }
}
