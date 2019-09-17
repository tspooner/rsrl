use crate::{Shared, geometry::{Matrix, MatrixView, MatrixViewMut}};
use super::*;

impl<X: ?Sized, T: StateFunction<X>> StateFunction<X> for Shared<T> {
    type Output = T::Output;

    fn evaluate(&self, state: &X) -> Self::Output { self.borrow().evaluate(state) }

    fn update(&mut self, state: &X, error: Self::Output) {
        self.borrow_mut().update(state, error)
    }
}

impl<X: ?Sized, T: DifferentiableStateFunction<X>> DifferentiableStateFunction<X> for Shared<T> {
    type Gradient = T::Gradient;

    fn grad(&self, state: &X) -> Self::Gradient { self.borrow().grad(state) }

    fn update_grad<G: crate::linalg::MatrixLike>(&mut self, grad: &G) {
        self.borrow_mut().update_grad(grad)
    }

    fn update_grad_scaled<G: crate::linalg::MatrixLike>(&mut self, grad: &G, factor: f64) {
        self.borrow_mut().update_grad_scaled(grad, factor)
    }
}

impl<X: ?Sized, U: ?Sized, T: StateActionFunction<X, U>> StateActionFunction<X, U> for Shared<T> {
    type Output = T::Output;

    fn evaluate(&self, state: &X, action: &U) -> Self::Output {
        self.borrow().evaluate(state, action)
    }

    fn update(&mut self, state: &X, action: &U, error: Self::Output) {
        self.borrow_mut().update(state, action, error)
    }
}

impl<X: ?Sized, U: ?Sized, T> DifferentiableStateActionFunction<X, U> for Shared<T>
where
    T: DifferentiableStateActionFunction<X, U>,
{
    type Gradient = T::Gradient;

    fn grad(&self, state: &X, action: &U) -> Self::Gradient { self.borrow().grad(state, action) }

    fn update_grad<G: crate::linalg::MatrixLike>(&mut self, grad: &G) {
        self.borrow_mut().update_grad(grad)
    }

    fn update_grad_scaled<G: crate::linalg::MatrixLike>(&mut self, grad: &G, factor: f64) {
        self.borrow_mut().update_grad_scaled(grad, factor)
    }
}

impl<X: ?Sized, T: EnumerableStateActionFunction<X>> EnumerableStateActionFunction<X> for Shared<T> {
    fn n_actions(&self) -> usize { self.borrow().n_actions() }

    fn evaluate_all(&self, state: &X) -> Vec<f64> { self.borrow().evaluate_all(state) }

    fn update_all(&mut self, state: &X, errors: Vec<f64>) {
        self.borrow_mut().update_all(state, errors)
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

    fn weights_dim(&self) -> [usize; 2] { self.borrow().weights_dim() }
}
