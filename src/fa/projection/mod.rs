use geometry::Space;
use geometry::norms::l1;
use ndarray::Array1;


#[derive(Clone, Serialize, Deserialize)]
pub enum Projection {
    Dense(Array1<f64>),
    Sparse(Array1<usize>),
}

impl Projection {
    pub fn z(&self) -> f64 {
        match self {
            &Projection::Dense(ref phi) => l1(phi.as_slice().unwrap()),
            &Projection::Sparse(ref indices) => indices.len() as f64,
        }
    }
}

impl Into<Projection> for Array1<f64> {
    fn into(self) -> Projection {
        Projection::Dense(self)
    }
}

impl Into<Projection> for Array1<usize> {
    fn into(self) -> Projection {
        Projection::Sparse(self)
    }
}


pub trait Projector<S: Space> {
    fn project(&self, input: &S::Repr) -> Projection;
    fn project_expanded(&self, input: &S::Repr) -> Array1<f64> {
        self.expand_projection(self.project(input))
    }

    fn dim(&self) -> usize;
    fn size(&self) -> usize;
    fn activation(&self) -> usize;
    fn equivalent(&self, other: &Self) -> bool;

    fn expand_projection(&self, projection: Projection) -> Array1<f64> {
        match projection {
            Projection::Dense(phi) => phi,
            Projection::Sparse(sparse_phi) => {
                let mut phi = Array1::zeros((self.size(),));

                for idx in sparse_phi.iter() {
                    phi[*idx] = 1.0;
                }

                phi
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

mod tile_coding;
pub use self::tile_coding::*;

mod uniform_grid;
pub use self::uniform_grid::*;
