//! Function approximation module.

use geometry::Space;

/// An interface for dealing with functions that may be evaluated.
pub trait Function<I: ?Sized, O> {
    /// Evaluates the function and returns its output.
    fn evaluate(&self, input: &I) -> O;
}

impl<I: ?Sized, O, T> Function<I, O> for Box<T>
where T: Function<I, O>
{
    fn evaluate(&self, input: &I) -> O { (**self).evaluate(input) }
}

/// An interface for dealing with adaptive functions.
pub trait Parameterised<I: ?Sized, U> {
    fn update(&mut self, input: &I, update: U);
}

impl<I: ?Sized, U, T> Parameterised<I, U> for Box<T>
where T: Parameterised<I, U>
{
    fn update(&mut self, input: &I, update: U) { (**self).update(input, update) }
}

/// An interface for state-value functions.
pub trait VFunction<S: Space>
    : Function<S::Repr, f64> + Parameterised<S::Repr, f64> {
    #[allow(unused_variables)]
    fn evaluate_phi(&self, phi: &Projection) -> f64 { unimplemented!() }

    #[allow(unused_variables)]
    fn update_phi(&mut self, phi: &Projection, update: f64) { unimplemented!() }
}

impl<S: Space, T> VFunction<S> for Box<T>
where T: VFunction<S>
{
    fn evaluate_phi(&self, phi: &Projection) -> f64 { (**self).evaluate_phi(phi) }

    fn update_phi(&mut self, phi: &Projection, update: f64) { (**self).update_phi(phi, update); }
}

/// An interface for action-value functions.
pub trait QFunction<S: Space>
    : Function<S::Repr, Vec<f64>> + Parameterised<S::Repr, Vec<f64>> {
    fn evaluate_action(&self, input: &S::Repr, action: usize) -> f64;
    fn update_action(&mut self, input: &S::Repr, action: usize, update: f64);

    #[allow(unused_variables)]
    fn evaluate_phi(&self, phi: &Projection) -> Vec<f64> {
        unimplemented!();
    }

    #[allow(unused_variables)]
    fn evaluate_action_phi(&self, phi: &Projection, action: usize) -> f64 {
        unimplemented!();
    }

    #[allow(unused_variables)]
    fn update_phi(&mut self, phi: &Projection, updates: Vec<f64>) {
        unimplemented!();
    }

    #[allow(unused_variables)]
    fn update_action_phi(&mut self, phi: &Projection, action: usize, update: f64) {
        unimplemented!();
    }
}

impl<S: Space, T> QFunction<S> for Box<T>
where T: QFunction<S>
{
    fn evaluate_action(&self, input: &S::Repr, action: usize) -> f64 {
        (**self).evaluate_action(input, action)
    }

    fn update_action(&mut self, input: &S::Repr, action: usize, update: f64) {
        (**self).update_action(input, action, update);
    }

    fn evaluate_phi(&self, phi: &Projection) -> Vec<f64> { (**self).evaluate_phi(phi) }

    fn evaluate_action_phi(&self, phi: &Projection, action: usize) -> f64 {
        (**self).evaluate_action_phi(phi, action)
    }

    fn update_phi(&mut self, phi: &Projection, updates: Vec<f64>) {
        (**self).update_phi(phi, updates);
    }

    fn update_action_phi(&mut self, phi: &Projection, action: usize, update: f64) {
        (**self).update_action_phi(phi, action, update)
    }
}

mod table;
pub use self::table::Table;

pub mod projection;
pub use self::projection::{Projection, Projector};

mod linear;
pub use self::linear::Linear;
