use geometry::Space;
use ndarray::Array1;


pub trait Projection<S: Space> {
    fn project_onto(&self, input: &S::Repr, phi: &mut Array1<f64>);
    fn project(&self, input: &S::Repr) -> Array1<f64> {
        let mut phi = Array1::zeros((self.size(),));
        self.project_onto(input, &mut phi);

        phi
    }

    fn dim(&self) -> usize;
    fn size(&self) -> usize;
    fn equivalent(&self, other: &Self) -> bool;
}

pub trait SparseProjection<S: Space>: Projection<S> {
    fn project_onto_sparse(&self, input: &S::Repr, indices: &mut Array1<usize>);
    fn project_sparse(&self, input: &S::Repr) -> Array1<usize> {
        let mut indices: Array1<usize> = Array1::zeros((self.sparsity(),));
        self.project_onto_sparse(input, &mut indices);

        indices
    }

    fn sparsity(&self) -> usize;
}

mod basis_network;
pub use self::basis_network::*;

mod rbf_network;
pub use self::rbf_network::*;

mod tile_coding;
pub use self::tile_coding::*;

mod uniform_grid;
pub use self::uniform_grid::*;
