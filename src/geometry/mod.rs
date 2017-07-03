mod span;
pub use self::span::Span;

mod spaces;
pub mod dimensions;
pub use self::spaces::*;

mod projection;
pub use self::projection::{Projection, project};


extern crate rusty_machine;
pub use self::rusty_machine::learning::toolkit::kernel as kernels;
