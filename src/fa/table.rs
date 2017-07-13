use super::{Function, Parameterised};

use std::ops::AddAssign;
use std::hash::Hash;
use std::collections::HashMap;

/// Generic tabular function representation.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use rsrl::fa::{Function, Parameterised};
/// use rsrl::fa::Table;
///
/// let f = {
///     let mut t = Table::<(u32, u32), f64>::new();
///     t.update(&(0, 1), 1.0);
///
///     t
/// };
///
/// assert_eq!(f.evaluate(&(0, 1)), 1.0);
/// ```
pub struct Table<K, V>(HashMap<K, V>);

impl<K: Hash + Eq, V> Table<K, V> {
    pub fn new() -> Self {
        Table(HashMap::new())
    }
}

// TODO: Have to deal with attempts to evaluate when no value is present.
//       Really we need a map with defaults.
//       The issue arises when we try to consider what the default value may be
//       for the generic type O.
impl<I, O> Function<I, O> for Table<I, O>
    where I: Hash + Eq,
          O: Copy + Default
{
    fn evaluate(&self, input: &I) -> O {
        if self.0.contains_key(input) {
            self.0[input]
        } else {
            O::default()
        }
    }
}

impl<I, E> Parameterised<I, E> for Table<I, E>
    where I: Hash + Eq + Copy,
          E: Clone + Default + AddAssign
{
    fn update(&mut self, input: &I, error: E) {
        *self.0.entry(*input).or_insert(E::default()) += error;
    }
}
