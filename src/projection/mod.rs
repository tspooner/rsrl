use ndarray::Array1;


pub trait Projection {
    fn project(&self, input: &[f64]) -> Array1<f64>;
    fn dim(&self) -> usize;
}

mod sutton_tc;
mod rbf_network;
mod uniform_grid;
mod basis_network;

pub use self::sutton_tc::*;
pub use self::rbf_network::*;
pub use self::uniform_grid::*;
pub use self::basis_network::*;
