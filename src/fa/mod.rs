// use std::iter::FromIterator;


pub trait Function<I: ?Sized, O> {
    fn evaluate(&self, input: &I) -> O;
    fn n_outputs(&self) -> usize;
}

pub trait Parameterised<I: ?Sized, E: ?Sized> {
    fn update(&mut self, input: &I, errors: &E);

    // TODO: Implement binary serialization with compression
    // fn load
    // fn save
}

macro_rules! add_support {
    ($ft:ty, Function, $it:ty, [$($ot:ty),+]) => {
        $(impl Function<$it, $ot> for $ft
            where $ft: Function<[f64], $ot>
        {
            fn evaluate(&self, inputs: &$it) -> $ot {
                self.evaluate(inputs.as_slice())
            }

            fn n_outputs(&self) -> usize {
                <Self as Function<[f64], $ot>>::n_outputs(self)
            }
        })+
    };
    ($ft:ty, Parameterised, $it:ty, [$($et:ty),+]) => {
        $(impl Parameterised<$it, $et> for $ft
            where $ft: Parameterised<[f64], $et>
        {
            fn update(&mut self, inputs: &$it, errors: &$et) {
                self.update(inputs.as_slice(), errors)
            }
        })+
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
