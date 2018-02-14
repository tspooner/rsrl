//! Linear basis projection module.

use Vector;
use geometry::Space;
use geometry::norms::l1;

/// Projected feature vector representation.
#[derive(Clone, Serialize, Deserialize)]
pub enum Projection {
    /// Dense, floating-point activation vector.
    Dense(Vector<f64>),

    /// Sparse, index-based activation vector.
    Sparse(Vector<usize>),
}

impl Projection {
    /// Compute the l1 normalisation constant of the projection.
    pub fn z(&self) -> f64 {
        match self {
            &Projection::Dense(ref phi) => l1(phi.as_slice().unwrap()),
            &Projection::Sparse(ref indices) => indices.len() as f64,
        }
    }
}

impl Into<Projection> for Vector<f64> {
    fn into(self) -> Projection { Projection::Dense(self) }
}

impl Into<Projection> for Vector<usize> {
    fn into(self) -> Projection { Projection::Sparse(self) }
}

/// Trait for basis projectors.
pub trait Projector<S: Space> {
    /// Project data from an input space onto the basis.
    fn project(&self, input: &S::Repr) -> Projection;

    /// Return the number of dimensions in the basis space.
    fn dim(&self) -> usize;

    /// Return the number of features in the basis space.
    fn size(&self) -> usize;

    /// Return the maximum number of active features in the basis space.
    fn activation(&self) -> usize;

    /// Check for equivalence with another projector of the same type.
    fn equivalent(&self, other: &Self) -> bool;

    /// Project data from an input space onto the basis and convert into a raw,
    /// dense vector.
    fn project_expanded(&self, input: &S::Repr) -> Vector<f64> {
        self.expand_projection(self.project(input))
    }

    /// Expand and normalise a given projection, and convert into a raw, dense
    /// vector.
    fn expand_projection(&self, projection: Projection) -> Vector<f64> {
        let z = match projection.z() {
            val if val.abs() < 1e-6 => 1.0,
            val => val,
        };

        match projection {
            Projection::Dense(phi) => phi / z,
            Projection::Sparse(sparse_phi) => {
                let mut phi = Vector::zeros((self.size(),));

                for idx in sparse_phi.iter() {
                    phi[*idx] = 1.0;
                }

                phi / z
            },
        }
    }
}

mod basis_network;
pub use self::basis_network::*;

mod rbf_network;
pub use self::rbf_network::*;

mod fourier;
pub use self::fourier::*;

mod polynomial;
pub use self::polynomial::*;

mod tile_coding;
pub use self::tile_coding::*;

mod uniform_grid;
pub use self::uniform_grid::*;
