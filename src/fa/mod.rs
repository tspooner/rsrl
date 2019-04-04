//! Function approximation and value function representation module.
use crate::{
    core::Shared,
    geometry::Vector,
};

extern crate lfa;
pub use self::lfa::{
    basis,
    core::*,
    eval,
    transforms,
    LFA,
    TransformedLFA,
};

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
