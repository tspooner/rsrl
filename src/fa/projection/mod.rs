use geometry::Space;
use ndarray::Array1;


pub trait Projection<S: Space> {
    fn project(&self, input: &S::Repr) -> Array1<f64>;
    fn dim(&self) -> usize;
    fn equivalent(&self, other: &Self) -> bool;
}

mod sutton_tc;
mod rbf_network;
mod uniform_grid;
mod basis_network;

pub use self::basis_network::*;
pub use self::rbf_network::*;
pub use self::sutton_tc::*;
pub use self::uniform_grid::*;
