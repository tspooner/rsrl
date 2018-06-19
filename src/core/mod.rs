#![allow(unused_variables)]
use domains::Transition;

pub trait Algorithm<S, A> {
    fn handle_sample(&mut self, sample: &Transition<S, A>);
    fn handle_terminal(&mut self, sample: &Transition<S, A>);
}

pub trait Controller<S, A>: Algorithm<S, A> {
    /// Sample the target policy for a given state `s`.
    fn pi(&mut self, s: &S) -> A;

    /// Sample the behaviour policy for a given state `s`.
    fn mu(&mut self, s: &S) -> A;
}

pub trait Predictor<S, A>: Algorithm<S, A> {
    fn v(&mut self, s: &S) -> f64 { unimplemented!() }
    fn qs(&mut self, s: &S) -> Vector<f64> { unimplemented!() }
    fn qsa(&mut self, s: &S, a: A) -> f64 { unimplemented!() }
}

mod memory;
pub use self::memory::*;

mod parameter;
pub use self::parameter::Parameter;

mod experiment;
pub use self::experiment::*;

mod trace;
pub use self::trace::*;

pub use geometry::{Matrix, Vector};
