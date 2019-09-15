use crate::{
    core::Shared,
    geometry::{Matrix, MatrixView, MatrixViewMut},
};
use super::*;

impl<S, T: Policy<S>> Policy<S> for Shared<T> {
    type Action = T::Action;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, state: &S) -> Self::Action {
        self.borrow().sample(rng, state)
    }

    fn mpa(&self, s: &S) -> Self::Action { self.borrow().mpa(s) }

    fn probability(&self, state: &S, a: &Self::Action) -> f64 {
        self.borrow().probability(state, a)
    }
}

impl<S, T: FinitePolicy<S>> FinitePolicy<S> for Shared<T> {
    fn n_actions(&self) -> usize { self.borrow().n_actions() }

    fn probabilities(&self, state: &S) -> Vec<f64> { self.borrow().probabilities(state) }
}

impl<S, T: DifferentiablePolicy<S>> DifferentiablePolicy<S> for Shared<T> {
    fn update(&mut self, state: &S, a: &Self::Action, error: f64) {
        self.borrow_mut().update(state, a, error)
    }

    fn update_grad(&mut self, grad: &MatrixView<f64>) {
        self.borrow_mut().update_grad(grad)
    }

    fn update_grad_scaled(&mut self, grad: &MatrixView<f64>, factor: f64) {
        self.borrow_mut().update_grad_scaled(grad, factor)
    }

    fn grad(&self, state: &S, a: &Self::Action) -> Matrix<f64> {
        self.borrow().grad(state, a)
    }

    fn grad_log(&self, state: &S, a: &Self::Action) -> Matrix<f64> {
        self.borrow().grad_log(state, a)
    }
}
