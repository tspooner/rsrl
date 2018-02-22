use super::{Approximator, EvaluationResult, UpdateResult};
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::AddAssign;

/// Generic tabular function representation.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use rsrl::fa::{Approximator, Table};
///
/// let f = {
///     let mut t = Table::<(u32, u32), f64>::new();
///     t.update(&(0, 1), 1.0);
///
///     t
/// };
///
/// assert_eq!(f.evaluate(&(0, 1)).unwrap(), 1.0);
/// ```
#[derive(Clone, Serialize, Deserialize)]
pub struct Table<K: Hash + Eq, V>(pub HashMap<K, V>);

impl<K: Hash + Eq, V> Table<K, V> {
    pub fn new() -> Self { Table(HashMap::new()) }
}

// TODO: Have to deal with attempts to evaluate when no value is present.
//       Really we need a map with defaults.
//       The issue arises when we try to consider what the default value may be
//       for the generic type O.
impl<I, V> Approximator<I> for Table<I, V>
where
    I: Hash + Eq + Copy,
    V: Copy + Default + AddAssign,
{
    type Value = V;

    fn evaluate(&self, input: &I) -> EvaluationResult<V> {
        if self.0.contains_key(input) {
            Ok(self.0[input])
        } else {
            Ok(V::default())
        }
    }

    fn update(&mut self, input: &I, error: V) -> UpdateResult<()> {
        *self.0.entry(*input).or_insert(V::default()) += error;

        Ok(())
    }
}
