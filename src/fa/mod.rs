use geometry::Space;

/// An interface for dealing with functions that may be evaluated.
pub trait Function<I: ?Sized, O> {
    /// Evaluates the function and returns its output.
    fn evaluate(&self, input: &I) -> O;
}

impl<I: ?Sized, O, T> Function<I, O> for Box<T>
    where T: Function<I, O>
{
    fn evaluate(&self, input: &I) -> O {
        (**self).evaluate(input)
    }
}


/// An interface for dealing with adaptive functions.
pub trait Parameterised<I: ?Sized, E> {
    fn update(&mut self, input: &I, error: E);
}

impl<I: ?Sized, E, T> Parameterised<I, E> for Box<T>
    where T: Parameterised<I, E>
{
    fn update(&mut self, input: &I, error: E) {
        (**self).update(input, error)
    }
}


/// An interface for value functions.
pub trait VFunction<S: Space>
    : Function<S::Repr, f64> + Parameterised<S::Repr, f64> {
    fn evaluate_phi(&self, _: &Projection) -> f64 {
        unimplemented!()
    }

    fn update_phi(&mut self, _: &Projection, _: f64) {
        unimplemented!()
    }
}

impl<S: Space, T> VFunction<S> for Box<T>
    where T: VFunction<S>
{
    fn evaluate_phi(&self, phi: &Projection) -> f64 {
        (**self).evaluate_phi(phi)
    }

    fn update_phi(&mut self, phi: &Projection, error: f64) {
        (**self).update_phi(phi, error);
    }
}


/// An interface for action-value functions.
pub trait QFunction<S: Space>
    : Function<S::Repr, Vec<f64>> + Parameterised<S::Repr, Vec<f64>> {
    fn evaluate_action(&self, input: &S::Repr, action: usize) -> f64;
    fn update_action(&mut self, input: &S::Repr, action: usize, error: f64);

    fn evaluate_phi(&self, _: &Projection) -> Vec<f64> {
        unimplemented!();
    }

    fn evaluate_action_phi(&self, _: &Projection, _: usize) -> f64 {
        unimplemented!();
    }

    fn update_phi(&mut self, _: &Projection, _: Vec<f64>) {
        unimplemented!();
    }

    fn update_action_phi(&mut self, _: &Projection, _: usize, _: f64) {
        unimplemented!();
    }
}

impl<S: Space, T> QFunction<S> for Box<T>
    where T: QFunction<S>
{
    fn evaluate_action(&self, input: &S::Repr, action: usize) -> f64 {
        (**self).evaluate_action(input, action)
    }

    fn update_action(&mut self, input: &S::Repr, action: usize, error: f64) {
        (**self).update_action(input, action, error);
    }

    fn evaluate_phi(&self, phi: &Projection) -> Vec<f64> {
        (**self).evaluate_phi(phi)
    }

    fn evaluate_action_phi(&self, phi: &Projection, action: usize) -> f64 {
        (**self).evaluate_action_phi(phi, action)
    }

    fn update_phi(&mut self, phi: &Projection, errors: Vec<f64>) {
        (**self).update_phi(phi, errors);
    }

    fn update_action_phi(&mut self, phi: &Projection, action: usize, error: f64) {
        (**self).update_action_phi(phi, action, error)
    }
}


mod table;
pub use self::table::Table;

pub mod projection;
pub use self::projection::{Projector, Projection};

mod linear;
pub use self::linear::Linear;
