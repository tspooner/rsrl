#![feature(test)]
extern crate test;

#[macro_use]
extern crate log;

#[macro_use]
#[feature(blas)]
extern crate ndarray;

// extern crate futures;
extern crate rand;

mod macros;

mod base_traits;
pub use base_traits::*;

pub mod loggers;
pub mod utils;
pub mod geometry;
pub mod domain;
pub mod fa;
pub mod policies;
pub mod agents;
