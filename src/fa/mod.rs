//! Function approximation and value function representation module.
#[cfg(test)]
pub(crate) mod mocking;

pub mod traces;

pub mod linear;
pub mod tabular;

pub mod transforms;
import_all!(transformed);

import_all!(shared);

pub use self::linear::{Parameterised, Weights, WeightsView, WeightsViewMut};

/// An interface for state value functions.
pub trait StateFunction<X: ?Sized> {
    type Output;

    fn evaluate(&self, state: &X) -> Self::Output;

    fn update(&mut self, state: &X, error: Self::Output);
}

pub trait DifferentiableStateFunction<X: ?Sized>: StateFunction<X> + Parameterised {
    type Gradient: crate::linalg::MatrixLike;

    fn grad(&self, state: &X) -> Self::Gradient;

    fn update_grad<G: crate::linalg::MatrixLike>(&mut self, grad: &G) {
        grad.addto(&mut self.weights_view_mut());
    }

    fn update_grad_scaled<G: crate::linalg::MatrixLike>(&mut self, grad: &G, factor: f64) {
        grad.scaled_addto(factor, &mut self.weights_view_mut());
    }
}

/// An interface for state-action value functions.
pub trait StateActionFunction<X: ?Sized, U: ?Sized> {
    type Output;

    fn evaluate(&self, state: &X, action: &U) -> Self::Output;

    fn update(&mut self, state: &X, action: &U, error: Self::Output);
}

pub trait DifferentiableStateActionFunction<X: ?Sized, U: ?Sized>: StateActionFunction<X, U> + Parameterised {
    type Gradient: crate::linalg::MatrixLike;

    fn grad(&self, state: &X, action: &U) -> Self::Gradient;

    fn update_grad<G: crate::linalg::MatrixLike>(&mut self, grad: &G) {
        grad.addto(&mut self.weights_view_mut());
    }

    fn update_grad_scaled<G: crate::linalg::MatrixLike>(&mut self, grad: &G, factor: f64) {
        grad.scaled_addto(factor, &mut self.weights_view_mut());
    }
}

pub trait EnumerableStateActionFunction<X: ?Sized>: StateActionFunction<X, usize, Output = f64> {
    fn n_actions(&self) -> usize;

    fn evaluate_all(&self, state: &X) -> Vec<f64>;

    fn update_all(&mut self, state: &X, errors: Vec<f64>);

    fn find_min(&self, state: &X) -> (usize, f64) {
        let mut iter = self.evaluate_all(state).into_iter().enumerate();
        let first = iter.next().unwrap();

        iter.fold(first, |acc, (i, x)| if acc.1 < x { acc } else { (i, x) })
    }

    fn find_max(&self, state: &X) -> (usize, f64) {
        let mut iter = self.evaluate_all(state).into_iter().enumerate();
        let first = iter.next().unwrap();

        iter.fold(first, |acc, (i, x)| if acc.1 > x { acc } else { (i, x) })
    }
}
