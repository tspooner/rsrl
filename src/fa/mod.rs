// use std::iter::FromIterator;
use ndarray::Array1;

/// An interface for dealing with functions that may be evaluated.
pub trait Function<I: ?Sized, O> {
    /// Evaluates the function for a given output.
    fn evaluate(&self, input: &I) -> O;

    /// Returns the number of values this function evaluates to.
    fn n_outputs(&self) -> usize;
}

/// An interface for dealing with adaptive functions that may be updated.
pub trait Parameterised<I: ?Sized, E: ?Sized> {
    fn update(&mut self, input: &I, errors: &E);
}

/// An interface for functions which have a linear representation.
pub trait Linear<I: ?Sized> {
    fn phi(&self, input: &I) -> Array1<f64>;
}


macro_rules! add_vec_support {
    ($ft:ty, Function, $($ot:ty),+) => {
        $(impl Function<Vec<f64>, $ot> for $ft
            where $ft: Function<[f64], $ot>
        {
            fn evaluate(&self, input: &Vec<f64>) -> $ot {
                self.evaluate(input.as_slice())
            }

            fn n_outputs(&self) -> usize {
                <Self as Function<[f64], $ot>>::n_outputs(self)
            }
        })+
    };
    ($ft:ty, Parameterised, $($et:ty),+) => {
        $(impl Parameterised<Vec<f64>, $et> for $ft
            where $ft: Parameterised<[f64], $et>
        {
            fn update(&mut self, input: &Vec<f64>, errors: &$et) {
                self.update(input.as_slice(), errors)
            }
        })+
    };
    ($ft:ty, Linear) => {
        impl Linear<Vec<f64>> for $ft where $ft: Linear<[f64]>
        {
            fn phi(&self, input: &Vec<f64>) -> Array1<f64> {
                self.phi(input.as_slice())
            }
        }
    }
}


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
