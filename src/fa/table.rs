use {Function, Parameterised};
use std::hash::Hash;
use std::collections::HashMap;

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

impl<I: Hash + Eq, O: Copy> Function<I, O> for Table<I, O> {
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
