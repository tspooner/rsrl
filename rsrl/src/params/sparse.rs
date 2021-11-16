use super::*;
use ndarray::{IntoDimension, Ix2};
use std::ops::{Add, AddAssign, Mul, MulAssign};

type GradMap = ::std::collections::HashMap<[usize; 2], f64>;

#[derive(Clone, Debug, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Sparse {
    dim: Ix2,
    grads: GradMap,
}

impl Sparse {
    pub fn new<D: IntoDimension<Dim = Ix2>>(dim: D, grads: GradMap) -> Result<Sparse, String> {
        if grads.is_empty() {
            Err("No gradient information passed into Sparse::new().".to_owned())
        } else {
            let dim = dim.into_dimension();
            let (nr, nc) = dim.into_pattern();

            if grads.keys().all(|&[r, c]| r < nr && c < nc) {
                Ok(Sparse { dim, grads })
            } else {
                Err("Inconsistent dimensions in Sparse::new().".to_owned())
            }
        }
    }

    pub fn new_unchecked<D: IntoDimension<Dim = Ix2>>(dim: D, grads: GradMap) -> Sparse {
        Sparse {
            dim: dim.into_dimension(),
            grads,
        }
    }
}

impl Buffer for Sparse {
    type Dim = Ix2;

    fn raw_dim(&self) -> Ix2 {
        self.dim
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

    fn to_dense(&self) -> Array2<f64> {
        let mut g_matrix = Array2::zeros(self.dim);

        for (&[r, c], &g) in self.grads.iter() {
            g_matrix[(r, c)] = g;
        }

        g_matrix
    }
}

impl BufferMut for Sparse {
    fn zeros<D: IntoDimension<Dim = Ix2>>(dim: D) -> Self {
        Self::new_unchecked(dim, GradMap::new())
    }

    fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Sparse {
            dim: self.dim,
            grads: self.grads.iter().map(|(&k, &pd)| (k, f(pd))).collect(),
        }
    }

    fn map_into(self, f: impl Fn(f64) -> f64) -> Self {
        Sparse {
            dim: self.dim,
            grads: self.grads.into_iter().map(|(k, pd)| (k, f(pd))).collect(),
        }
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.grads.values_mut().for_each(|pd| *pd = f(*pd));
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        if self.dim != other.dim {
            panic!("Incompatible dimensionality.")
        }

        for (k, x) in self.grads.iter_mut() {
            *x = f(*x, other.grads.get(k).copied().unwrap_or(0.0));
        }

        for (&k, y) in other.grads.iter() {
            self.grads.entry(k).or_insert_with(|| f(0.0, *y));
        }
    }
}

impl Into<Array2<f64>> for Sparse {
    fn into(self) -> Array2<f64> {
        self.to_dense()
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
                self.grads.insert(idx, *pd_other);
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
