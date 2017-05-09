extern crate libc;
use self::libc::{c_double, c_int, size_t};

use super::{Function, Parameterised, Linear, QFunction};

use ndarray::{Axis, Array1};
use geometry::{Space, RegularSpace};
use geometry::dimensions::Continuous;


#[link(name="tiles", kind="static")]
extern {
    fn tiles(tile_indices: *mut size_t, nt: c_int, memory: c_int,
             floats: *const c_double, nf: c_int,
             ints: *const c_int, ni: c_int);
}


pub struct SuttonTiles {
    weights: Array1<c_double>,

    n_outputs: usize,
    n_tilings: i32,
    memory_size: i32,

    int_array: [c_int; 1]
}

impl SuttonTiles {
    pub fn new(n_tilings: i32, memory_size: i32, n_outputs: usize) -> Self {
        SuttonTiles {
            weights: Array1::<c_double>::zeros(memory_size as usize),

            n_outputs: n_outputs,
            n_tilings: n_tilings,
            memory_size: memory_size,

            int_array: [0],
        }
    }

    fn load_tiles(&self, floats: &[c_double], ints: &[c_int]) -> Vec<size_t> {
        let mut ti = vec![0; self.n_tilings as usize];

        unsafe {
            tiles(ti.as_mut_ptr(), self.n_tilings, self.memory_size,
                  floats.as_ptr(), floats.len() as c_int,
                  ints.as_ptr(), ints.len() as c_int);
        }

        ti
    }

    fn evaluate_index(&self, input: &Vec<f64>, index: c_int) -> f64 {
        let tiles = self.load_tiles(input, &[index]);

        self.weights.select(Axis(0), tiles.as_slice()).scalar_sum() / self.n_tilings as f64
    }
}


impl Function<Vec<f64>, f64> for SuttonTiles {
    fn evaluate(&self, input: &Vec<f64>) -> f64 {
        self.evaluate_index(input, 0)
    }
}

impl Function<Vec<f64>, Vec<f64>> for SuttonTiles {
    fn evaluate(&self, input: &Vec<f64>) -> Vec<f64> {
        (0..self.n_outputs).map(|c| self.evaluate_index(input, c as c_int)).collect()
    }
}


impl Parameterised<Vec<f64>, f64> for SuttonTiles {
    fn update(&mut self, input: &Vec<f64>, error: f64) {
        self.int_array[0] = 0;

        for t in self.load_tiles(input, &self.int_array) {
            self.weights[t] += error;
        }
    }
}

impl Parameterised<Vec<f64>, Vec<f64>> for SuttonTiles {
    fn update(&mut self, input: &Vec<f64>, errors: Vec<f64>) {
        for c in 0..self.n_outputs {
            self.int_array[0] = c as c_int;

            for t in self.load_tiles(input, &self.int_array) {
                self.weights[t] += errors[c];
            }
        }
    }
}


// TODO: Implement Linear - problem is that phi will be a function of the state
//       and action for this implementation of tile coding.


impl QFunction<RegularSpace<Continuous>> for SuttonTiles
{
    fn evaluate_action(&self, input: &Vec<f64>, action: usize) -> f64 {
        self.evaluate_index(input, action as c_int)
    }

    fn update_action(&mut self, input: &Vec<f64>, action: usize, error: f64) {
        self.int_array[0] = action as c_int;

        for t in self.load_tiles(input, &self.int_array) {
            self.weights[t] += error;
        }
    }
}


#[cfg(test)]
mod tests {
    use super::SuttonTiles;

    use fa::{Function, Parameterised};
    use geometry::RegularSpace;
    use geometry::dimensions::Partition;

    #[test]
    fn test_simple() {
        let mut t = SuttonTiles::new(1, 1000, 1);

        t.update(&vec![1.5], 25.5);

        let out: f64 = t.evaluate(&vec![1.5]);
        assert_eq!(out, 25.5);

        t.update(&vec![1.5], -12.75);

        let out: f64 = t.evaluate(&vec![1.5]);
        assert_eq!(out, 12.75);
    }
}
