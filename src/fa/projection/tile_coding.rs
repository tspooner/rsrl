extern crate libc;
use self::libc::{c_double, c_int, size_t};

use super::Projection;
use super::tchs::RNDSEQ;
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
pub struct HashTileCoding {
    n_tilings: u32,
    memory_size: usize,
}

impl HashTileCoding {
    pub fn new(n_tilings: u32, memory_size: usize) -> Self {
        HashTileCoding {
            n_tilings: n_tilings,
            memory_size: memory_size,
        }
    }

    fn hash_unh(coordinates: &[i32], m: usize, inc: i32) -> usize {
        let sum = coordinates.iter().enumerate().fold(0, |acc, (i, c)| {
            let mut index = c + inc*(i as i32);
            while index < 0 {
                index += 2048;
            }

            acc + RNDSEQ[(index & 2047) as usize]
        });

        let mut index = sum % m;
        while index < 0 {
            index += m;
        }

        index
    }
}

impl Projection<RegularSpace<Continuous>> for HashTileCoding {
    fn project_onto(&self, input: &Vec<f64>, phi: &mut Array1<f64>) {
        let n_floats = input.len();
        let n_tilings = self.n_tilings as i32;

        let mut base = vec![0; n_floats];
        let mut coordinates = vec![0; n_floats+1];

        let qstate: Vec<i32> = input.iter().map(|f| {
            (f*(n_tilings as f64)).floor() as i32
        }).collect();

        for t in 0..n_tilings {
            coordinates[n_floats] = t;

            for i in 0..n_floats {
                if qstate[i] >= base[i] {
                    coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % n_tilings);
                } else {
                    coordinates[i] = qstate[i]+1 + ((base[i] - qstate[i] - 1) % n_tilings) - n_tilings;
                }

                base[i] += 1 + (2 * i as i32);
            }

            phi[HashTileCoding::hash_unh(&coordinates, self.memory_size, 449)] = 1.0;
        }
    }

    fn dim(&self) -> usize {
        self.memory_size as usize
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.dim() == other.dim() && self.n_tilings == other.n_tilings &&
        self.memory_size == other.memory_size
    }
}



#[derive(Serialize, Deserialize)]
pub struct SuttonTileCoding {
    n_tilings: i32,
    memory_size: i32,

    int_array: [i32; 0],
}

impl SuttonTileCoding {
    pub fn new(n_tilings: i32, memory_size: i32) -> Self {
        SuttonTileCoding {
            n_tilings: n_tilings,
            memory_size: memory_size,

            int_array: [],
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

impl Projection<RegularSpace<Continuous>> for SuttonTileCoding {
    fn project_onto(&self, input: &Vec<f64>, phi: &mut Array1<f64>) {
        for i in self.load_tiles(input, &self.int_array) {
            phi[i] = 1.0;
        }
    }

    fn dim(&self) -> usize {
        self.memory_size as usize
    }

    fn equivalent(&self, other: &Self) -> bool {
        self.dim() == other.dim() && self.n_tilings == other.n_tilings &&
        self.memory_size == other.memory_size
    }
}
