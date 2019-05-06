//! Function approximation and value function representation module.
use crate::{
    core::Shared,
    geometry::{Vector, Matrix, MatrixView, MatrixViewMut},
};

pub use lfa::{
    basis,
    composition::Composable,
    core::*,
    transforms,
    TransformedLFA,
    LFA
};

mod macros;

#[cfg(test)]
pub(crate) mod mocking;

/// An interface for state-value functions.
pub trait VFunction<S: ?Sized>: Embedding<S> + ScalarApproximator {
    fn state_value(&self, s: &S) -> f64 {
        self.evaluate(&self.embed(s)).unwrap()
    }
}

impl<S: ?Sized, T: Embedding<S> + ScalarApproximator<Output = f64>> VFunction<S> for T {}

/// An interface for action-value functions.
pub trait QFunction<S: ?Sized>: Embedding<S> + VectorApproximator {
    fn action_values(&self, s: &S) -> Vector<f64> {
        self.evaluate(&self.embed(s)).unwrap()
    }

    fn action_value(&self, s: &S, a: usize) -> f64 {
        self.action_values(s)[a]
    }
}

impl<S: ?Sized, T: Embedding<S> + VectorApproximator> QFunction<S> for T {}

// Shared<T> impls:
impl<S: ?Sized, T: Embedding<S>> Embedding<S> for Shared<T> {
    fn n_features(&self) -> usize {
        self.borrow().n_features()
    }

    fn embed(&self, s: &S) -> Features {
        self.borrow().embed(s)
    }
}

impl<T: Approximator> Approximator for Shared<T> {
    type Output = T::Output;

    fn n_outputs(&self) -> usize { self.borrow().n_outputs() }

    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
        self.borrow().evaluate(features)
    }

    fn jacobian(&self, features: &Features) -> Matrix<f64> {
        self.borrow().jacobian(features)
    }

    fn update_grad(&mut self, grad: &Matrix<f64>, update: Self::Output) -> UpdateResult<()> {
        self.borrow_mut().update_grad(grad, update)
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
}
