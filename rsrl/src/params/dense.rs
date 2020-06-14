use super::*;
use ndarray::{Array1, Array2, ArrayBase, Data, DataMut, Ix1, Ix2, IntoDimension};
use std::ops::AddAssign;

pub type Vector = Array1<f64>;
pub type Matrix = Array2<f64>;

impl<D: Data<Elem = f64>> Buffer for ArrayBase<D, Ix1> {
    type Dim = Ix1;

    fn raw_dim(&self) -> Ix1 { self.raw_dim() }

    fn addto<DM: DataMut<Elem = f64>>(&self, weights: &mut ArrayBase<DM, Ix1>) {
        weights.add_assign(self);
    }

    fn scaled_addto<DM: DataMut<Elem = f64>>(&self, alpha: f64, weights: &mut ArrayBase<DM, Ix1>) {
        weights.scaled_add(alpha, self);
    }

    fn to_dense(&self) -> Array1<f64> { self.to_owned() }
}

impl BufferMut for Array1<f64> {
    fn zeros<D: IntoDimension<Dim = Ix1>>(dim: D) -> Self { Self::zeros(dim) }

    fn map(&self, f: impl Fn(f64) -> f64) -> Self { self.mapv(f) }

    fn map_into(self, f: impl Fn(f64) -> f64) -> Self { self.mapv_into(f) }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) { self.mapv_inplace(f); }

    fn merge(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        // TODO: this can be implemented much more efficiently.
        self.clone().merge_into(other, f)
    }

    fn merge_into(mut self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        self.merge_inplace(other, f);
        self
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        self.zip_mut_with(other, |x, y| *x = f(*x, *y));
    }
}

impl<D: Data<Elem = f64>> Buffer for ArrayBase<D, Ix2> {
    type Dim = Ix2;

    fn raw_dim(&self) -> Ix2 { self.raw_dim() }

    fn addto<DM: DataMut<Elem = f64>>(&self, weights: &mut ArrayBase<DM, Ix2>) {
        weights.add_assign(self);
    }

    fn scaled_addto<DM: DataMut<Elem = f64>>(&self, alpha: f64, weights: &mut ArrayBase<DM, Ix2>) {
        weights.scaled_add(alpha, self);
    }

    fn to_dense(&self) -> Array2<f64> { self.to_owned() }
}

impl BufferMut for Array2<f64> {
    fn zeros<D: IntoDimension<Dim = Ix2>>(dim: D) -> Self { Self::zeros(dim) }

    fn map(&self, f: impl Fn(f64) -> f64) -> Self { self.mapv(f) }

    fn map_into(self, f: impl Fn(f64) -> f64) -> Self { self.mapv_into(f) }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) { self.mapv_inplace(f); }

    fn merge(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        // TODO: this can be implemented much more efficiently.
        self.clone().merge_into(other, f)
    }

    fn merge_into(mut self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        self.merge_inplace(other, f);
        self
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        self.zip_mut_with(other, |x, y| *x = f(*x, *y));
    }
}
