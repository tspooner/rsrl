//! Agent policy module.

// TODO: Add support for generic action spaces representation.
pub trait Policy {
    fn sample(&mut self, qs: &[f64]) -> usize;
    fn probabilities(&mut self, qs: &[f64]) -> Vec<f64>;

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
