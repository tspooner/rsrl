//! Agent policy module.

/// Policy trait for functions that select between a set of values.
pub trait Policy<I: ?Sized, A> {
    /// Sample the policy distribution for a given input.
    fn sample(&mut self, input: &I) -> A;

    /// Return the probability of selecting an action for a given input.
    fn probability(&mut self, input: &I, a: A) -> f64;

    /// Update the policy after reaching a terminal state.
    fn handle_terminal(&mut self) {}
}

pub trait FinitePolicy<I: ?Sized>: Policy<I, usize> {
    /// Return the probability of selecting each action for a given input.
    fn probabilities(&mut self, input: &I) -> Vec<f64>;
}

mod random;
pub use self::random::Random;

mod greedy;
pub use self::greedy::Greedy;

mod epsilon_greedy;
pub use self::epsilon_greedy::EpsilonGreedy;

mod boltzmann;
pub use self::boltzmann::Boltzmann;

mod truncated_boltzmann;
pub use self::truncated_boltzmann::TruncatedBoltzmann;
