use {Function, Parameterised};
use std::hash::Hash;
use std::collections::HashMap;

pub struct Table<K, V>(HashMap<K, V>);

impl<K: Hash + Eq, V> Table<K, V> {
    pub fn new() -> Self {
        Table(HashMap::new())
    }
}

impl<I: Hash + Eq, O: Copy> Function<I, O> for Table<I, O> {
    fn evaluate(&self, input: &I) -> O {
        self.0[input]
    }
}

impl<I: Hash + Eq + Copy, T: Copy> Parameterised<I, T> for Table<I, T> {
    fn update(&mut self, input: &I, errors: &T) {
        self.0.insert(*input, *errors);
    }
}
