#![allow(unused_variables)]
use domains::{Transition, Observation::Terminal};
use geometry::Vector;

pub trait Algorithm {
    /// Perform housekeeping after terminal state observation.
    fn handle_terminal(&mut self) {}
}

pub trait OnlineLearner<S, A>: Algorithm {
    /// Handle a single transition collected from the problem environment.
    fn handle_transition(&mut self, transition: &Transition<S, A>);

    /// Handle an arbitrary sequence of transitions collected from the problem environment.
    fn handle_sequence(&mut self, sequence: &[Transition<S, A>]) {
        sequence.into_iter().for_each(|ref t| self.handle_transition(t));
    }
}

pub trait BatchLearner<S, A>: Algorithm {
    /// Handle a batch of samples collected from the problem environment.
    fn handle_batch(&mut self, batch: &[Transition<S, A>]);
}

pub trait Controller<S, A> {
    /// Sample the target policy for a given state `s`.
    fn sample_target(&mut self, s: &S) -> A;

    /// Sample the behaviour policy for a given state `s`.
    fn sample_behaviour(&mut self, s: &S) -> A;
}

pub trait ValuePredictor<S> {
    /// Compute the estimated value of V(s).
    fn predict_v(&mut self, s: &S) -> f64;
}

pub trait ActionValuePredictor<S, A>: ValuePredictor<S> {
    /// Compute the estimated value of Q(s, a).
    fn predict_qsa(&mut self, s: &S, a: A) -> f64 {
        self.predict_v(s)
    }

    /// Compute the estimated value of Q(s, ·).
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        unimplemented!()
    }
}
