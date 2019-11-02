use crate::linalg::{MatrixLike, Entry};
use super::*;
use std::ops::{Add, AddAssign, Mul, MulAssign};

type GradMap = ::std::collections::HashMap<[usize; 2], f64>;

#[derive(Clone, Debug, Default)]
pub struct Sparse {
    dim: [usize; 2],
    grads: GradMap,
}

impl Sparse {
    pub fn new(dim: [usize; 2], grads: GradMap) -> Sparse {
        if grads.len() == 0 {
            panic!("No gradient information passed into Sparse::new().");
        } else {
            if grads.keys().all(|&[r, c]| r < dim[0] && c < dim[1]) {
                Sparse {
                    dim,
                    grads,
                }
            } else {
                panic!("Inconsistent dimensions in Sparse::new().");
            }
        }
    }

    pub unsafe fn new_unchecked(dim: [usize; 2], grads: GradMap) -> Sparse {
        Sparse { dim, grads }
    }
}

impl MatrixLike for Sparse {
    fn zeros(dim: [usize; 2]) -> Self {
        unsafe { Self::new_unchecked(dim, GradMap::new()) }
    }

    fn dim(&self) -> [usize; 2] { self.dim }

    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Sparse {
            dim: self.dim,
            grads: self.grads.into_iter().map(|(k, pd)| (k, f(pd))).collect(),
        }
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.grads.values_mut().for_each(|pd| *pd = f(*pd));
    }

    fn combine_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        for (k, x) in self.grads.iter_mut() {
            *x = f(*x, other.grads.get(k).cloned().unwrap_or(0.0));
        }

        for (&k, y) in other.grads.iter() {
            self.grads.entry(k).or_insert_with(|| f(0.0, *y));
        }
    }

    fn for_each(&self, mut f: impl FnMut(Entry)) {
        self.grads.iter().map(|(index, gradient)| Entry {
            index: *index,
            gradient: *gradient,
        }).for_each(|pd| f(pd));
    }

    fn addto<D: DataMut<Elem = f64>>(&self, weights: &mut ArrayBase<D, Ix2>) {
        for (&idx, pd) in self.grads.iter() {
            if let Some(w) = weights.get_mut(idx) {
                w.add_assign(pd);
            }
        }
    }

    fn scaled_addto<D: DataMut<Elem = f64>>(&self, alpha: f64, weights: &mut ArrayBase<D, Ix2>) {
        for (&idx, &pd) in self.grads.iter() {
            if let Some(w) = weights.get_mut(idx) {
                *w = w.mul_add(alpha, pd);
            }
        }
    }
}

impl Into<Array2<f64>> for Sparse {
    fn into(self) -> Array2<f64> {
        let mut g_matrix = Array2::zeros((self.dim[0], self.dim[1]));

        for ([r, c], g) in self.grads.into_iter() {
            g_matrix[(r, c)] = g;
        }

        g_matrix
    }
}

impl Add<Sparse> for Sparse {
    type Output = Sparse;

    fn add(mut self, other: Sparse) -> Sparse {
        self.add_assign(&other);

        self
    }
}

impl AddAssign<Sparse> for Sparse {
    fn add_assign(&mut self, other: Sparse) {
        for (idx, pd_other) in other.grads.into_iter() {
            if let Some(pd) = self.grads.get_mut(&idx) {
                pd.add_assign(&pd_other);
            } else {
                self.grads.insert(idx, pd_other);
            }
        }
    }
}

impl AddAssign<&Sparse> for Sparse {
    fn add_assign(&mut self, other: &Sparse) {
        for (&idx, pd_other) in other.grads.iter() {
            if let Some(pd) = self.grads.get_mut(&idx) {
                pd.add_assign(pd_other);
            } else {
                self.grads.insert(idx, pd_other.clone());
            }
        }
    }
}

impl Mul<f64> for Sparse {
    type Output = Sparse;

    fn mul(mut self, factor: f64) -> Sparse {
        self.mul_assign(factor);

        self
    }
}

impl MulAssign<f64> for Sparse {
    fn mul_assign(&mut self, factor: f64) {
        for pd in self.grads.values_mut() {
            pd.mul_assign(factor);
        }
    }
}
