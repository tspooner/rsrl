#[allow(unused_imports)]
#[macro_use]
extern crate ndarray;
extern crate rand;

#[macro_use]
extern crate slog;
extern crate slog_async;
extern crate slog_term;

extern crate serde;
#[macro_use]
extern crate serde_derive;

pub extern crate spaces as geometry;
pub use self::geometry::{Matrix, Vector};

mod core;
pub use self::core::*;

mod parameter;
pub use self::parameter::Parameter;

mod experiment;
pub use self::experiment::*;

mod consts;
mod macros;

pub mod utils;
pub mod logging;

pub mod domains;
pub use self::domains::{Transition, Observation};

pub mod agents;
pub mod fa;
pub mod policies;
