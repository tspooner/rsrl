// #![feature(test)]
// extern crate test;

#[macro_use]
extern crate log;

#[macro_use]
#[feature(blas)]
extern crate ndarray;

// extern crate futures;
extern crate rand;

mod utils;
mod macros;

mod experiment;
pub use self::experiment::*;

pub mod geometry;
pub mod domain;
pub mod fa;
pub mod policies;
pub mod agents;
pub mod logging;
