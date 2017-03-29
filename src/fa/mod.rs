// use std::iter::FromIterator;

// pub trait LinearFA {
    // fn phi<O: FromIterator<f64>>(&self, inputs: &[f64]) -> O;
// }


mod table;
pub use self::table::Table;

mod partitions;
pub use self::partitions::Partitions;

mod rbf_network;
pub use self::rbf_network::RBFNetwork;

mod basis_network;
pub use self::basis_network::{BasisFunction, BasisNetwork};

// pub mod leaf;

// mod sutton_tiles;
// pub use self::sutton_tiles::SuttonTiles;
