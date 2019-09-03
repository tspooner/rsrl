use crate::core::Shared;
use super::{Features, LinearStateFunction, LinearStateActionFunction};

impl<X: ?Sized, T: LinearStateFunction<X>> LinearStateFunction<X> for Shared<T> {
    fn n_features(&self) -> usize { self.borrow().n_features() }

    fn features(&self, state: &X) -> Features { self.borrow().features(state) }

    fn evaluate_features(&self, features: &Features) -> f64 {
        self.borrow().evaluate_features(features)
    }

    fn update_features(&mut self, features: &Features, error: f64) {
        self.borrow_mut().update_features(features, error)
    }
}

impl<X: ?Sized, U: ?Sized, T> LinearStateActionFunction<X, U> for Shared<T>
where
    T: LinearStateActionFunction<X, U>,
{
    fn n_features(&self) -> usize { self.borrow().n_features() }

    fn features(&self, state: &X, action: &U) -> Features {
        self.borrow().features(state, action)
    }

    fn evaluate_features(&self, features: &Features, action: &U) -> f64 {
        self.borrow().evaluate_features(features, action)
    }

    fn update_features(&mut self, features: &Features, action: &U, error: f64) {
        self.borrow_mut().update_features(features, action, error)
    }
}
