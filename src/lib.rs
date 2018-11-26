#![allow(unused_imports)]

#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate rand;

#[macro_use]
extern crate slog;
extern crate slog_async;
extern crate slog_term;

extern crate serde;
#[macro_use]
extern crate serde_derive;

pub extern crate spaces as geometry;

mod macros;
mod utils;

pub(crate) mod consts;
pub mod control;
pub mod core;
pub mod domains;
pub mod fa;
pub mod logging;
pub mod policies;
pub mod prediction;
