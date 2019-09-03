use crate::geometry::{Vector, Matrix, MatrixViewMut};
use std::ops::{AddAssign, MulAssign};

pub struct PartialDerivative {
    pub index: [usize; 2],
    pub gradient: f64,
}

pub trait Gradient: Clone + Into<Matrix<f64>> {
    fn dim(&self) -> [usize; 2];

    fn map(mut self, f: impl Fn(f64) -> f64) -> Self {
        self.map_inplace(f); self
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64);

    fn combine(mut self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        self.combine_inplace(other, f); self
    }

    fn combine_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64);

    fn for_each(&self, f: impl FnMut(PartialDerivative));

    fn addto(&self, weights: &mut MatrixViewMut<f64>) {
        self.for_each(|pd| weights[pd.index] += pd.gradient);
    }

    fn scaled_addto(&self, alpha: f64, weights: &mut MatrixViewMut<f64>) {
        self.for_each(|pd| weights[pd.index] += alpha * pd.gradient);
    }

    fn to_matrix(&self) -> Matrix<f64> { self.clone().into() }
}

import_all!(matrix);
// import_all!(column);
import_all!(columnar);
import_all!(sparse);

// import_all!(arithmetic);
// import_all!(scaling);
// import_all!(shifting);
