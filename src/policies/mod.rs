// TODO: Add support for generic action spaces representation.
pub trait Policy {
    fn sample(&mut self, qs: &[f64]) -> usize;
    fn handle_terminal(&mut self) {}
}


pub mod random;

mod greedy;
pub use self::greedy::Greedy;

mod epsilon_greedy;
pub use self::epsilon_greedy::EpsilonGreedy;

mod boltzmann;
pub use self::boltzmann::Boltzmann;
