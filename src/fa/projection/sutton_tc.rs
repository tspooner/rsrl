extern crate libc;
use self::libc::{c_double, c_int, size_t};

use super::Projection;
use geometry::RegularSpace;
use geometry::dimensions::Continuous;
use ndarray::Array1;


#[link(name="tiles", kind="static")]
extern "C" {
    fn tiles(tile_indices: *mut size_t,
             nt: c_int,
             memory: c_int,
             floats: *const c_double,
             nf: c_int,
             ints: *const c_int,
             ni: c_int);
}


#[derive(Serialize, Deserialize)]
pub struct SuttonTiles {
    n_tilings: i32,
    memory_size: i32,

    int_array: [i32; 1],
}

impl SuttonTiles {
    pub fn new(n_tilings: i32, memory_size: i32, int_offset: i32) -> Self {
        SuttonTiles {
            n_tilings: n_tilings,
            memory_size: memory_size,
            int_array: [int_offset],
        }
    }

    fn load_tiles(&self, floats: &[c_double], ints: &[c_int]) -> Vec<size_t> {
        let mut ti = vec![0; self.n_tilings as usize];

        unsafe {
            tiles(ti.as_mut_ptr(),
                  self.n_tilings,
                  self.memory_size,
                  floats.as_ptr(),
                  floats.len() as c_int,
                  ints.as_ptr(),
                  ints.len() as c_int);
        }

        ti
    }
}

impl Projection<RegularSpace<Continuous>> for SuttonTiles {
    // TODO: Rework how we handle cases where we need a sparse vector. There's
    // no point dealing with memory_size of floats!
    fn project(&self, input: &Vec<f64>) -> Array1<f64> {
        let mut p = Array1::<f64>::zeros(self.memory_size as usize);

        for i in self.load_tiles(input, &self.int_array) {
            p[i] = 1.0;
        }

        p
    }

    fn dim(&self) -> usize {
        self.memory_size as usize
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.dim() == other.dim() && self.n_tilings == other.n_tilings &&
        self.memory_size == other.memory_size
    }
}
