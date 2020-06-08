use crate::Shared;
use ndarray::{Array, Array2, ArrayBase, DataMut, Dimension};

pub trait Buffer: Sized {
    type Dim: Dimension;

    fn dim(&self) -> <Self::Dim as Dimension>::Pattern;

    fn addto<E: DataMut<Elem = f64>>(&self, arr: &mut ArrayBase<E, Self::Dim>) {
        self.scaled_addto(1.0, arr)
    }

    fn scaled_addto<E: DataMut<Elem = f64>>(&self, alpha: f64, arr: &mut ArrayBase<E, Self::Dim>);

    fn to_dense(&self) -> Array<f64, Self::Dim> {
        let mut arr = Array::zeros(self.dim());

        self.addto(&mut arr);

        arr
    }

    fn into_dense(self) -> Array<f64, Self::Dim> { self.to_dense() }
}

impl<T: Buffer> Buffer for &T {
    type Dim = T::Dim;

    fn dim(&self) -> <Self::Dim as Dimension>::Pattern { (*self).dim() }

    fn addto<E: DataMut<Elem = f64>>(&self, arr: &mut ArrayBase<E, Self::Dim>) {
        (*self).addto(arr)
    }

    fn scaled_addto<E: DataMut<Elem = f64>>(&self, alpha: f64, arr: &mut ArrayBase<E, Self::Dim>) {
        (*self).scaled_addto(alpha, arr)
    }

    fn to_dense(&self) -> Array<f64, Self::Dim> { (*self).to_dense() }

    fn into_dense(self) -> Array<f64, Self::Dim> { self.to_dense() }
}

pub trait BufferMut: Buffer + Clone {
    fn zeros(dim: <Self::Dim as Dimension>::Pattern) -> Self;

    fn map(&self, f: impl Fn(f64) -> f64) -> Self;

    fn map_into(self, f: impl Fn(f64) -> f64) -> Self {
        self.map(f);
        self
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64);

    fn merge(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self;

    fn merge_into(self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        self.merge(other, f);
        self
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64);
}

mod columnar;
mod dense;
mod sparse;
mod tile;

pub use self::columnar::Columnar;
pub use self::sparse::Sparse;
pub use self::tile::Tile;

/// Matrix populated with _owned_ weights.
pub type Weights = ndarray::Array2<f64>;

/// Matrix populated with _referenced_ weights.
pub type WeightsView<'a> = ndarray::ArrayView2<'a, f64>;

/// Matrix populated with _mutably referenced_ weights.
pub type WeightsViewMut<'a> = ndarray::ArrayViewMut2<'a, f64>;

/// Types that are parameterised by a matrix of weights.
pub trait Parameterised {
    /// Return an owned copy of the weights.
    fn weights(&self) -> Weights { self.weights_view().to_owned() }

    /// Return a read-only view of the weights.
    fn weights_view(&self) -> WeightsView;

    /// Return a mutable view of the weights.
    fn weights_view_mut(&mut self) -> WeightsViewMut;

    /// Return the dimensions of the weight matrix.
    fn weights_dim(&self) -> (usize, usize) { self.weights_view().dim() }

    fn n_weights(&self) -> usize {
        let (r, c) = self.weights_dim();

        r * c
    }
}

impl<F: Parameterised> Parameterised for Shared<F> {
    fn weights(&self) -> Weights { self.borrow().weights() }

    fn weights_view(&self) -> WeightsView {
        unsafe { self.as_ptr().as_ref().unwrap().weights_view() }
    }

    fn weights_view_mut(&mut self) -> WeightsViewMut {
        unsafe { self.as_ptr().as_mut().unwrap().weights_view_mut() }
    }

    fn weights_dim(&self) -> (usize, usize) { self.borrow().weights_dim() }
}
