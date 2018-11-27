#![allow(unused_variables)]
use domains::Transition;

pub trait Algorithm<S, A> {
    fn handle_sample(&mut self, sample: &Transition<S, A>);
    fn handle_terminal(&mut self, sample: &Transition<S, A>);
}

pub trait BatchAlgorithm<S, A>: Algorithm<S, A> {
    fn solve(&mut self);

    fn handle_batch(&mut self, samples: &Vec<Transition<S, A>>) {
        use domains::Observation;

        samples.into_iter().rev().for_each(|s| {
            match s.to {
                Observation::Terminal(_) => self.handle_terminal(s),
                _ => self.handle_sample(s),
            }
        });

        self.solve();
    }
}

pub trait Controller<S, A>: Algorithm<S, A> {
    /// Sample the target policy for a given state `s`.
    fn sample_target(&mut self, s: &S) -> A;

    /// Sample the behaviour policy for a given state `s`.
    fn sample_behaviour(&mut self, s: &S) -> A;
}

pub trait Predictor<S, A>: Algorithm<S, A> {
    fn predict_v(&mut self, s: &S) -> f64 { unimplemented!() }
    fn predict_qs(&mut self, s: &S) -> Vector<f64> { unimplemented!() }
    fn predict_qsa(&mut self, s: &S, a: A) -> f64 { self.predict_v(s) }
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
