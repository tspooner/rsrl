//! Reinforcement learning should be _fast_, _safe_ and _easy to use_.
//!
//! `rsrl` provides generic constructs for reinforcement learning (RL) experiments in an extensible
//! framework with efficient implementations of existing methods for rapid prototyping.
#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate rand;
extern crate rand_distr;

#[macro_use]
extern crate slog;
extern crate slog_async;
extern crate slog_term;

#[macro_use]
extern crate serde;

extern crate lfa;

mod macros;
mod utils;

pub(crate) mod consts;

import_all!(memory);
import_all!(parameter);
import_all!(experiment);
import_all!(deref_slice);

pub mod domains;
pub mod logging;

pub mod linalg;
pub extern crate spaces;

#[macro_use]
pub mod fa;
pub mod prediction;
pub mod policies;
pub mod control;

pub trait Algorithm {
    /// Perform housekeeping after terminal state observation.
    fn handle_terminal(&mut self) {}
}

impl<T: Algorithm> Algorithm for Shared<T> {
    fn handle_terminal(&mut self) {
        self.borrow_mut().handle_terminal()
    }
}

pub trait OnlineLearner<S, A>: Algorithm {
    /// Handle a single transition collected from the problem environment.
    fn handle_transition(&mut self, transition: &domains::Transition<S, A>);

    /// Handle an arbitrary sequence of transitions collected from the problem environment.
    fn handle_sequence(&mut self, sequence: &[domains::Transition<S, A>]) {
        sequence.into_iter().for_each(|ref t| self.handle_transition(t));
    }
}

impl<S, A, T: OnlineLearner<S, A>> OnlineLearner<S, A> for Shared<T> {
    fn handle_transition(&mut self, transition: &domains::Transition<S, A>) {
        self.borrow_mut().handle_transition(transition)
    }

    fn handle_sequence(&mut self, sequence: &[domains::Transition<S, A>]) {
        self.borrow_mut().handle_sequence(sequence)
    }
}

pub trait BatchLearner<S, A>: Algorithm {
    /// Handle a batch of samples collected from the problem environment.
    fn handle_batch(&mut self, batch: &[domains::Transition<S, A>]);
}

impl<S, A, T: BatchLearner<S, A>> BatchLearner<S, A> for Shared<T> {
    fn handle_batch(&mut self, batch: &[domains::Transition<S, A>]) {
        self.borrow_mut().handle_batch(batch)
    }
}
