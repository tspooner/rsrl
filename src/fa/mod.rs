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


/// An interface for linearly parameterised functions.
pub trait Linear<S: Space> {
    fn phi(&self, input: &S::Repr) -> Array1<f64>;
}


/// An interface for value functions.
pub trait VFunction<S: Space>:
    Function<S::Repr, f64> + Parameterised<S::Repr, f64>
{
    fn evaluate_phi(&self, _: &Array1<f64>) -> f64 {
        unimplemented!()
    }

    fn update_phi(&mut self, _: &Array1<f64>, _: f64) {
        unimplemented!()
    }
}

impl<S: Space, T> VFunction<S> for Box<T>
    where T: VFunction<S>
{
    fn evaluate_phi(&self, phi: &Array1<f64>) -> f64 {
        (**self).evaluate_phi(phi)
    }

    fn update_phi(&mut self, phi: &Array1<f64>, error: f64) {
        (**self).update_phi(phi, error);
    }
}


/// An interface for action-value functions.
pub trait QFunction<S: Space>:
    Function<S::Repr, Vec<f64>> + Parameterised<S::Repr, Vec<f64>>
{
    fn evaluate_action(&self, input: &S::Repr, action: usize) -> f64;
    fn update_action(&mut self, input: &S::Repr, action: usize, error: f64);

    fn evaluate_phi(&self, _: &Array1<f64>) -> Vec<f64> {
        unimplemented!();
    }

    fn evaluate_action_phi(&self, _: &Array1<f64>, _: usize) -> f64 {
        unimplemented!();
    }

    fn update_phi(&mut self, _: &Array1<f64>, _: Vec<f64>) {
        unimplemented!();
    }

    fn update_action_phi(&mut self, _: &Array1<f64>, _: usize, _: f64) {
        unimplemented!();
    }

    // NOTE: Eventually we may want to distinguish between the feature vector
    //       for each action. I can't see this being needed anytime soon so
    //       for now we will just have a single entry point for computing phi.
    // fn phi(&self, input: &S::Repr) -> Array2<f64> {
        // unimplemented!()
    // }

    // fn phi_action(&self, input: &S::Repr, _: usize) -> Array1<f64>{
        // unimplemented!()
    // }

    // fn evaluate_phi(&self, phi: &Array2<f64>) -> f64 {
        // unimplemented!();
    // }

    // fn evaluate_action_phi(&self, phi: &Array1<f64>, action: usize) -> f64 {
        // unimplemented!();
    // }

    // fn update_phi(&mut self, phi: &Array2<f64>, error: f64) -> f64 {
        // unimplemented!();
    // }

    // fn update_action_phi(&mut self, phi: &Array1<f64>, action: usize, error: f64) -> f64 {
        // unimplemented!();
    // }
}

impl<S: Space, T> QFunction<S> for Box<T>
    where T: QFunction<S>
{
    fn evaluate_action(&self, input: &S::Repr, action: usize) -> f64 {
        (**self).evaluate_action(input, action)
    }

    fn update_action(&mut self, input: &S::Repr, action: usize, error: f64) {
        (**self).update_action(input, action, error);
    }

    fn evaluate_phi(&self, phi: &Array1<f64>) -> Vec<f64> {
        (**self).evaluate_phi(phi)
    }

    fn evaluate_action_phi(&self, phi: &Array1<f64>, action: usize) -> f64 {
        (**self).evaluate_action_phi(phi, action)
    }

    fn update_phi(&mut self, phi: &Array1<f64>, error: Vec<f64>) {
        (**self).update_phi(phi, error);
    }

    fn update_action_phi(&mut self, phi: &Array1<f64>, action: usize, error: f64) {
        (**self).update_action_phi(phi, action, error)
    }
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
