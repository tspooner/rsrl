use ndarray::Array1;
use geometry::Space;

/// An interface for dealing with functions that may be evaluated.
pub trait Function<I: ?Sized, O> {
    /// Evaluates the function and returns its output.
    fn evaluate(&self, input: &I) -> O;
}

impl<I: ?Sized, O, T> Function<I, O> for Box<T>
    where T: Function<I, O>
{
    fn evaluate(&self, input: &I) -> O {
        (**self).evaluate(input)
    }
}


/// An interface for dealing with adaptive functions.
pub trait Parameterised<I: ?Sized, E> {
    fn update(&mut self, input: &I, error: E);
}

impl<I: ?Sized, E, T> Parameterised<I, E> for Box<T>
    where T: Parameterised<I, E>
{
    fn update(&mut self, input: &I, error: E) {
        (**self).update(input, error)
    }
}


/// An interface for functions which have a linear representation.
pub trait Linear<I: ?Sized> {
    fn phi(&self, input: &I) -> Array1<f64>;
}

impl<I: ?Sized, T> Linear<I> for Box<T>
    where T: Linear<I>
{
    fn phi(&self, input: &I) -> Array1<f64> {
        (**self).phi(input)
    }
}


macro_rules! add_vec_support {
    ($ft:ty, Function, $($ot:ty),+) => {
        $(impl Function<Vec<f64>, $ot> for $ft {
            fn evaluate(&self, input: &Vec<f64>) -> $ot {
                <Self as Function<[f64], $ot>>::evaluate(self, input.as_slice())
            }
        })+
    };
    ($ft:ty, Parameterised, $($et:ty),+) => {
        $(impl Parameterised<Vec<f64>, $et> for $ft {
            fn update(&mut self, input: &Vec<f64>, error: $et) {
                <Self as Parameterised<[f64], $et>>::update(self, input.as_slice(), error);
            }
        })+
    };
    ($ft:ty, Linear) => {
        impl Linear<Vec<f64>> for $ft {
            fn phi(&self, input: &Vec<f64>) -> Array1<f64> {
                <Self as Linear<[f64]>>::phi(self, input.as_slice())
            }
        }
    }
}


pub trait VFunction<S: Space>:
    Function<S::Repr, f64> + Parameterised<S::Repr, f64> {}

impl<S: Space, T> VFunction<S> for T
    where T: Function<S::Repr, f64> + Parameterised<S::Repr, f64> {}


pub trait QFunction<S: Space>:
    Function<S::Repr, Vec<f64>> + Parameterised<S::Repr, Vec<f64>>
{
    fn evaluate_action(&self, input: &S::Repr, action: usize) -> f64;
    fn update_action(&mut self, input: &S::Repr, action: usize, error: f64);
}


// XXX: Bug with Rust that renders ICE errors when using VFunctionGroup.
mod fgroup;
pub use self::fgroup::VFunctionGroup;


mod table;
mod partitions;
mod rbf_network;
mod basis_network;
// mod leaf;
// mod sutton_tiles;

pub mod exact {
    pub use fa::table::*;
}

pub mod linear {
    pub use fa::partitions::*;
    pub use fa::rbf_network::*;
    pub use fa::basis_network::*;
    // pub use fa::sutton_tiles::*;
}

pub mod non_linear {
    // pub use fa::leaf::*;
}
