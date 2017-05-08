extern crate libc;
use self::libc::{c_int, c_double};

use super::{Function, Parameterised, Linear, VFunction, QFunction};

use ndarray::{Array1, Array2};
use geometry::{Space, RegularSpace};
use geometry::dimensions::Continuous;


#[link(name="tiles", kind="static")]
extern {
    fn tiles(tile_indices: *mut c_int, nt: c_int, memory: c_int,
             floats: *const c_double, nf: c_int,
             ints: *const c_int, ni: c_int);
}


pub struct SuttonTiles {
    weights: Array2<c_double>,

    n_tilings: i32,
    memory_size: i32,

    int_array: [c_int; 1]
}

impl SuttonTiles {
    pub fn new(n_tilings: i32, memory_size: i32, n_outputs: usize) -> Self {
        SuttonTiles {
            weights: Array2::<c_double>::zeros((memory_size as usize, n_outputs)),

            n_tilings: n_tilings,
            memory_size: memory_size,

            int_array: [0],
        }
    }

    fn load_tiles(&self, floats: &[c_double], ints: &[c_int]) -> Vec<i32> {
        let mut ti = vec![0; self.n_tilings as usize];

        unsafe {
            tiles(ti.as_mut_ptr(), self.n_tilings, self.memory_size,
                  floats.as_ptr(), floats.len() as c_int,
                  ints.as_ptr(), ints.len() as c_int);
        }

        ti
    }

    fn evaluate_column(&self, input: &Vec<f64>, column: usize) -> f64 {
        self.load_tiles(input, &[column as c_int]).iter().fold(0.0, |acc, &i| {
            acc + self.weights[[i as usize, column]]
        }) / self.n_tilings as f64
    }
}


impl Function<Vec<f64>, f64> for SuttonTiles
{
    fn evaluate(&self, input: &Vec<f64>) -> f64 {
        self.evaluate_column(input, 0)
    }
}

impl Function<Vec<f64>, Vec<f64>> for SuttonTiles
{
    fn evaluate(&self, input: &Vec<f64>) -> Vec<f64> {
        (0..self.weights.cols()).map(|c| self.evaluate_column(input, c)).collect()
    }
}


impl Parameterised<Vec<f64>, f64> for SuttonTiles {
    fn update(&mut self, input: &Vec<f64>, error: f64) {
        self.int_array[0] = 0;

        for r in self.load_tiles(input, &self.int_array) {
            self.weights[[r as usize, 0]] += error;
        }
    }
}

impl Parameterised<Vec<f64>, Vec<f64>> for SuttonTiles
{
    fn update(&mut self, input: &Vec<f64>, errors: Vec<f64>) {
        for c in 0..self.weights.cols() {
            self.int_array[0] = c as c_int;

            for r in self.load_tiles(input, &self.int_array) {
                self.weights[[r as usize, c]] += errors[c];
            }
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
