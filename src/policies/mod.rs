//! Agent policy module.

// TODO: Add support for generic action spaces representation.
/// Policy trait for functions that select between a set of values.
pub trait Policy {
    /// Sample the policy distribution for a set of exogenous input values.
    fn sample(&mut self, qs: &[f64]) -> usize;

    /// Return the probability of selecting each value in a given set of input values.
    fn probabilities(&mut self, qs: &[f64]) -> Vec<f64>;

    /// Update the policy after reaching a terminal state.
    fn handle_terminal(&mut self) {}
}


mod random;
pub use self::random::Random;

mod greedy;
pub use self::greedy::Greedy;

mod epsilon_greedy;
pub use self::epsilon_greedy::EpsilonGreedy;

mod boltzmann;
pub use self::boltzmann::{Boltzmann, TruncatedBoltzmann};
