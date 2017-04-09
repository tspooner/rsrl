use {Function, Parameterised};
use std::hash::Hash;
use std::collections::HashMap;

/// Generic tabular function representation.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use rsrl::fa::{Function, Parameterised, Table};
///
/// let f = {
///     let mut t = Table::<(u32, u32), f64>::new(1);
///     t.update(&(0, 1), &1.0);
///
///     t
/// };
///
/// assert_eq!(f.evaluate(&(0, 1)), 1.0);
/// ```
pub struct Table<K, V> {
    mapping: HashMap<K, V>,

    n_outputs: usize,
}

impl<K: Hash + Eq, V> Table<K, V> {
    pub fn new(n_outputs: usize) -> Self {
        Table {
            mapping: HashMap::new(),

            n_outputs: n_outputs
        }
    }
}

// TODO: Have to deal with attempts to evaluate when no value is present.
//       Really we need a map with defaults.
//       The issue arises when we try to consider what the default value may be
//       for the generic type O.
impl<I: Hash + Eq + Clone, O: Copy> Function<I, O> for Table<I, O> {
    fn evaluate(&self, input: &I) -> O {
        self.mapping[input]
    }

    fn n_outputs(&self) -> usize {
        self.n_outputs
    }
}

impl<I: Hash + Eq + Copy, T: Copy> Parameterised<I, T> for Table<I, T> {
    fn update(&mut self, input: &I, errors: &T) {
        self.mapping.insert(*input, *errors);
    }
}
