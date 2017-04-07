// #![feature(test)]
// extern crate test;

#[macro_use]
extern crate log;

#[macro_use]
#[feature(blas)]
extern crate ndarray;

// extern crate futures;
extern crate rand;

mod macros;

pub mod utils;
pub mod geometry;
pub mod domain;

pub mod fa;
pub use fa::{Function, Parameterised};

pub mod policies;
pub mod agents;
pub mod experiment;
pub mod loggers;
