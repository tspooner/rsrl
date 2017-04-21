use super::{Function, Parameterised, Linear};

use ndarray::{Array1, Array2};


#[link(name="tiles")]
extern {
    fn tiles(tile_indices: *mut i32, nt: i32, memory: i32,
             floats: *const f64, nf: i32, h1: i32);
}


pub struct SuttonTiles {
    weights: Array2<f64>,

    n_tilings: i32,
    memory_size: i32,
}

impl SuttonTiles {
    pub fn new(n_tilings: i32, memory_size: i32, n_outputs: usize) -> Self {
        SuttonTiles {
            weights: Array2::<f64>::zeros((memory_size as usize, n_outputs)),

            n_tilings: n_tilings,
            memory_size: memory_size,
        }
    }

    fn load_tiles(&self, input: &[f64], h: i32) -> Vec<i32> {
        let mut ti = vec![0; self.n_tilings as usize];

        unsafe {
            tiles(ti.as_mut_ptr(), self.n_tilings, self.memory_size,
                  input.as_ptr(), input.len() as i32, h);
        }

        ti
    }
}


impl Function<[f64], Vec<f64>> for SuttonTiles {
    fn evaluate(&self, input: &[f64]) -> Vec<f64> {
        let evaluate_column = |c| {
            self.load_tiles(input, c as i32).iter().fold(0.0, |acc, &i| {
                acc + self.weights[[i as usize, c]]
            }) / self.n_tilings as f64
        };

        (0..self.weights.cols()).map(|c| evaluate_column(c)).collect()
    }

    fn n_outputs(&self) -> usize {
        self.weights.cols()
    }
}

add_vec_support!(SuttonTiles, Function, Vec<f64>);


impl Parameterised<[f64], [f64]> for SuttonTiles {
    fn update(&mut self, input: &[f64], errors: &[f64]) {
        for c in 0..self.weights.cols() {
            for r in self.load_tiles(input, c as i32) {
                self.weights[[r as usize, c]] += errors[c];
            }
        }
    }
}

add_vec_support!(SuttonTiles, Parameterised, [f64]);


// impl Linear<[f64]> for SuttonTiles {
    // fn phi(&self, input: &[f64]) -> Array1<f64> {
    // }
// }

// add_vec_support!(SuttonTiles, Linear);
