#[allow(unused_imports)]
#[macro_use]
extern crate ndarray;
extern crate rand;

#[macro_use]
extern crate slog;
extern crate slog_async;
extern crate slog_term;

extern crate blas;
extern crate openblas_src;

extern crate serde;
#[macro_use]
extern crate serde_derive;

mod consts;
mod macros;
pub mod utils;

pub extern crate spaces as geometry;
pub use self::geometry::{Matrix, Vector};

mod parameter;
pub use self::parameter::Parameter;

pub mod agents;
pub mod domains;
pub mod fa;
pub mod logging;
pub mod policies;

mod experiment;
pub use self::experiment::*;
