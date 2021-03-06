use crate::Shared;
use ndarray::{Array, Array2, ArrayBase, DataMut, Dimension, IntoDimension};

/// Gradient buffer with arbitrary dimension.
pub trait Buffer: Sized {
    /// Dimensionality od the the buffer.
    type Dim: Dimension;

    /// Return the dimensionality of the `Buffer` in pattern matching-friendly form.
    fn dim(&self) -> <Self::Dim as Dimension>::Pattern { self.raw_dim().into_pattern() }

    /// Return the number of dimensions of the `Buffer`.
    fn n_dim(&self) -> usize { self.raw_dim().ndim() }

    /// Return the dimensionality of the `Buffer`.
    fn raw_dim(&self) -> Self::Dim;

    /// Add the buffer's state to a mutable tensor of equal dimensionality.
    fn addto<E: DataMut<Elem = f64>>(&self, arr: &mut ArrayBase<E, Self::Dim>) {
        self.scaled_addto(1.0, arr)
    }

    /// Add the buffer's state (scaled) to a mutable tensor of equal dimensionality.
    ///
    /// This is a common operation in SGD-type methods and can typically be implemented in a highly
    /// optimised form compared to a pair of addition/scale mutations.
    fn scaled_addto<E: DataMut<Elem = f64>>(&self, alpha: f64, arr: &mut ArrayBase<E, Self::Dim>);

    /// Construct a dense tensor representation from the `Buffer` state.
    fn to_dense(&self) -> Array<f64, Self::Dim> {
        let mut arr = Array::zeros(self.dim());

        self.addto(&mut arr);

        arr
    }

    /// Convert the `Buffer` into a dense tensor.
    fn into_dense(self) -> Array<f64, Self::Dim> { self.to_dense() }
}

impl<T: Buffer> Buffer for &T {
    type Dim = T::Dim;

    fn dim(&self) -> <Self::Dim as Dimension>::Pattern { (*self).dim() }

    fn n_dim(&self) -> usize { (*self).n_dim() }

    fn raw_dim(&self) -> Self::Dim { (*self).raw_dim() }

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
    fn zeros<D: IntoDimension<Dim = Self::Dim>>(dim: D) -> Self;

    fn reset(&mut self) { self.map_inplace(|_| 0.0) }

    fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        self.clone().map_into(f)
    }

    fn map_into(mut self, f: impl Fn(f64) -> f64) -> Self {
        self.map_inplace(f);
        self
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64);

    fn merge(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        self.clone().merge_into(other, f)
    }

    fn merge_into(mut self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        self.merge_inplace(other, f);
        self
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64);
}

mod dense;
mod sparse;

mod columnar;
mod tile;

pub use self::{
    dense::{Vector, Matrix},
    sparse::Sparse,

    columnar::Columnar,
    tile::Tile,
};

/// Matrix populated with _owned_ weights.
pub type Weights = Matrix;

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
