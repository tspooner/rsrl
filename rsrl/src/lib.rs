//! Reinforcement learning should be _fast_, _safe_ and _easy to use_.
//!
//! `rsrl` provides generic constructs for reinforcement learning (RL)
//! experiments in an extensible framework with efficient implementations of
//! existing methods for rapid prototyping.
#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate rand;
extern crate rand_distr;

#[cfg_attr(feature = "serde", macro_use)]
#[cfg(feature = "serde")]
extern crate serde_crate;

#[allow(unused_imports)]
#[macro_use]
extern crate rsrl_derive;
#[doc(hidden)]
pub use self::rsrl_derive::*;

extern crate lfa;
pub extern crate rsrl_domains as domains;

mod core;
mod utils;
pub use self::core::*;

pub extern crate spaces;

pub mod params;
#[macro_use]
pub mod fa;
pub mod control;
pub mod policies;
pub mod prediction;
pub mod traces;
