#![allow(unused_imports)]

#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate rand;
extern crate rand_distr;

#[macro_use]
extern crate slog;
extern crate slog_async;
extern crate slog_term;

#[macro_use]
extern crate serde;

#[macro_use]
extern crate lfa;

pub mod geometry {
    extern crate spaces;

    pub use self::spaces::*;

    /// 1d array type.
    pub type Vector<T = f64> = ndarray::Array1<T>;
    pub type VectorView<'a, T = f64> = ndarray::ArrayView1<'a, T>;
    pub type VectorViewMut<'a, T = f64> = ndarray::ArrayViewMut1<'a, T>;

    /// 2d array type.
    pub type Matrix<T = f64> = ndarray::Array2<T>;
    pub type MatrixView<'a, T = f64> = ndarray::ArrayView2<'a, T>;
    pub type MatrixViewMut<'a, T = f64> = ndarray::ArrayViewMut2<'a, T>;
}

mod macros;
mod utils;

pub(crate) mod consts;
pub mod core;
pub mod linalg;
pub mod domains;
pub mod logging;

#[macro_use]
pub mod fa;
pub mod prediction;
pub mod policies;
pub mod control;
