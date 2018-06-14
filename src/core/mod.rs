#![allow(unused_variables)]
use domains::Transition;

pub trait Handler<SAMPLE> {
    #[allow(unused_variables)]
    fn handle_sample(&mut self, sample: &SAMPLE) {}

    #[allow(unused_variables)]
    fn handle_terminal(&mut self, sample: &SAMPLE) {}
}

pub trait BatchHandler<SAMPLE>: Handler<SAMPLE> {
    fn handle_batch(&mut self, batch: &Vec<SAMPLE>) {
        for sample in batch.into_iter() {
            self.handle_sample(sample);
        }
    }
}

pub trait Controller<S, A>: Handler<Transition<S, A>> {
    /// Sample the target policy for a given state `s`.
    fn pi(&mut self, s: &S) -> A;

    /// Sample the behaviour policy for a given state `s`.
    fn mu(&mut self, s: &S) -> A;
}

pub trait Predictor<S, A>: Handler<Transition<S, A>> {
    fn predict_v(&mut self, s: &S) -> f64;

    fn predict_qs(&mut self, s: &S) -> Vector<f64> { unimplemented!() }
    fn predict_qsa(&mut self, s: &S, a: A) -> f64 { unimplemented!() }
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
