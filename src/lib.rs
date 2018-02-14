#[macro_use]
extern crate slog;
extern crate slog_async;
extern crate slog_term;

extern crate blas;
extern crate openblas_src;

#[macro_use]
extern crate ndarray;

extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate serde_test;

extern crate rand;

pub mod utils;
mod consts;
mod macros;

pub extern crate spaces as geometry;
pub use self::geometry::{Vector, Matrix};

mod parameter;
pub use self::parameter::Parameter;

pub mod agents;
pub mod domains;
pub mod fa;
pub mod logging;
pub mod policies;

mod experiment;
pub use self::experiment::*;
