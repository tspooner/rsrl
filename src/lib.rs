#[macro_use] extern crate slog;
extern crate slog_term;
extern crate slog_async;

#[macro_use]
#[feature(blas)]
extern crate ndarray;

extern crate serde;
extern crate serde_json;
extern crate serde_test;
#[macro_use] extern crate serde_derive;

// extern crate futures;
extern crate rand;

#[macro_use] extern crate itertools;
use itertools::Itertools;

mod utils;
mod consts;
mod macros;

pub mod parameter;
pub use self::parameter::Parameter;

mod experiment;
pub use self::experiment::*;

pub mod agents;
pub mod domains;
pub mod fa;
pub mod geometry;
pub mod logging;
pub mod policies;
