//! Geometry module.

pub mod norms;

mod span;
pub use self::span::Span;

/// 1d array type.
pub type Vector<T = f64> = super::ndarray::Array1<T>;

/// 2d array type.
pub type Matrix<T = f64> = super::ndarray::Array2<T>;

mod spaces;
pub mod dimensions;
pub use self::spaces::*;

extern crate rusty_machine;
pub use self::rusty_machine::learning::toolkit::kernel as kernels;
