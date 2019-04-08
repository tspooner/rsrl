//! Function approximation and value function representation module.
use crate::{
    core::Shared,
    geometry::{Vector, Matrix, MatrixView, MatrixViewMut},
};

extern crate lfa;
pub use self::lfa::{basis, core::*, eval, transforms, TransformedLFA, LFA};

#[cfg(test)]
pub(crate) mod mocking;

// mod table;
// pub use self::table::Table;

/// An interface for state-value functions.
pub trait VFunction<S: ?Sized>: Embedded<S> + ScalarApproximator {}

impl<S: ?Sized, T: Embedded<S> + ScalarApproximator<Output = f64>> VFunction<S> for T {}

/// An interface for action-value functions.
pub trait QFunction<S: ?Sized>: Embedded<S> + VectorApproximator {}

impl<S: ?Sized, T: Embedded<S> + VectorApproximator> QFunction<S> for T {}

// Shared<T> impls:
impl<S: ?Sized, T: Embedded<S>> Embedded<S> for Shared<T> {
    fn n_features(&self) -> usize {
        self.borrow().n_features()
    }

    fn to_features(&self, s: &S) -> Features {
        self.borrow().to_features(s)
    }
}

impl<T: Approximator> Approximator for Shared<T> {
    type Output = T::Output;

    fn n_outputs(&self) -> usize { self.borrow().n_outputs() }

    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
        self.borrow().evaluate(features)
    }

    fn update(&mut self, features: &Features, update: Self::Output) -> UpdateResult<()> {
        self.borrow_mut().update(features, update)
    }
}

impl<T: Parameterised> Parameterised for Shared<T> {
    fn weights(&self) -> Matrix<f64> { self.borrow().weights() }

    fn weights_view(&self) -> MatrixView<f64> {
        unsafe { self.as_ptr().as_ref().unwrap().weights_view() }
    }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> {
        unsafe { self.as_ptr().as_mut().unwrap().weights_view_mut() }
    }

    fn weights_dim(&self) -> (usize, usize) { self.borrow().weights_dim() }

    fn weights_count(&self) -> usize { self.borrow().weights_count() }
}
