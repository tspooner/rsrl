#![allow(unused_variables)]
use domains::Transition;

pub trait Algorithm<S, A> {
    fn handle_transition(&mut self, transition: &Transition<S, A>);
    fn handle_terminal(&mut self, sample: &S);
}

pub trait BatchAlgorithm<S, A>: Algorithm<S, A> {
    fn solve(&mut self);

    fn handle_batch(&mut self, transitions: &Vec<Transition<S, A>>) {
        use domains::Observation::Terminal;

        transitions.into_iter().rev().for_each(|t| {
            self.handle_transition(t);

            if let Terminal(s) = t.to {
                self.handle_terminal(s);
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
