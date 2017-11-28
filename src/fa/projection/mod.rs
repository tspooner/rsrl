use geometry::Space;
use ndarray::Array1;


pub trait Projection<S: Space> {
    fn project_onto(&self, input: &S::Repr, phi: &mut Array1<f64>);
    fn project(&self, input: &S::Repr) -> Array1<f64> {
        let mut phi = Array1::zeros((self.dim(),));
        self.project_onto(input, &mut phi);

        phi
    }

    fn dim(&self) -> usize;
    fn equivalent(&self, other: &Self) -> bool;
}

mod basis_network;
pub use self::basis_network::*;

mod rbf_network;
pub use self::rbf_network::*;

mod tchs;
mod tile_coding;
pub use self::tile_coding::*;

mod uniform_grid;
pub use self::uniform_grid::*;
