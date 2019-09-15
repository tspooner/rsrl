use crate::{
    geometry::{Vector, Matrix, MatrixViewMut},
    linalg::{MatrixLike, Entry},
};
use ndarray::Array1;
use std::{
    collections::HashMap,
    ops::{Add, AddAssign, Mul, MulAssign},
};
use super::{Features, dot_features};

#[derive(Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct LFAGradient {
    pub dim: [usize; 2],
    features: HashMap<usize, Features>,
}

impl LFAGradient {
    pub fn new(dim: [usize; 2], features: HashMap<usize, Features>) -> Self {
        LFAGradient { dim, features, }
    }

    pub fn from_features(dim: [usize; 2], column: usize, features: Features) -> Self {
        let mut hm = HashMap::new();
        hm.insert(column, features);

        LFAGradient::new(dim, hm)
    }

    pub fn features(&self, column: &usize) -> Option<&Features> { self.features.get(column) }

    pub fn dot(&self, other: &Self) -> HashMap<usize, f64> {
        self.features.iter()
            .filter_map(|(a, f1)| {
                other.features.get(a).map(|f2| (a, f1, f2))
            })
            .map(|(a, f1, f2)| (*a, dot_features(f1, f2)))
            .collect()
    }
}

impl MatrixLike for LFAGradient {
    fn zeros(dim: [usize; 2]) -> Self {
        Self::new(dim, HashMap::new())
    }

    fn dim(&self) -> [usize; 2] { self.dim }

    fn map(mut self, f: impl Fn(f64) -> f64) -> Self {
        for pds in self.features.values_mut() {
            pds.mut_activations(&f);
        }

        self
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.features.values_mut().for_each(|pds| pds.mut_activations(&f));
    }

    fn combine_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        for (c, xs) in self.features.iter_mut() {
            if let Some(ys) = other.features.get(c) {
                let mut local_xs = unsafe {
                    ::std::mem::replace(xs, Features::Dense(Array1::uninitialized(0)))
                }.combine(ys, &f);

                ::std::mem::swap(&mut local_xs, xs);
            } else {
                xs.mut_activations(|x| f(x, 0.0));
            }
        }

        for (&c, ys) in other.features.iter() {
            self.features.entry(c).or_insert_with(|| {
                let mut ys = ys.clone();
                ys.mut_activations(|y| f(0.0, y));
                ys
            });
        }
    }

    fn for_each(&self, mut f: impl FnMut(Entry)) {
        for (&c, features) in self.features.iter() {
            match features {
                Features::Dense(activations) => {
                    activations.iter().enumerate().for_each(|(i, &a)| f(Entry {
                        index: [i, c],
                        gradient: a,
                    }))
                },
                Features::Sparse(_, activations) => {
                    activations.iter().for_each(|(&i, &a)| f(Entry {
                        index: [i, c],
                        gradient: a,
                    }))
                },
            }
        }
    }

    fn addto(&self, weights: &mut MatrixViewMut<f64>) {
        for (&c, features) in self.features.iter() {
            features.addto(&mut weights.column_mut(c));
        }
    }

    fn scaled_addto(&self, alpha: f64, weights: &mut MatrixViewMut<f64>) {
        for (&c, features) in self.features.iter() {
            features.scaled_addto(alpha, &mut weights.column_mut(c));
        }
    }
}

impl Into<Matrix<f64>> for LFAGradient {
    fn into(self) -> Matrix<f64> {
        let mut g_matrix = Matrix::zeros((self.dim[0], self.dim[1]));

        for (c, f) in self.features.into_iter() {
            f.addto(&mut g_matrix.column_mut(c));
        }

        g_matrix
    }
}

impl AddAssign<&LFAGradient> for LFAGradient {
    fn add_assign(&mut self, other: &LFAGradient) {
        for (&c, pds_other) in other.features.iter() {
            if let Some(pds) = self.features.get_mut(&c) {
                let mut local_pds = unsafe {
                    ::std::mem::replace(pds, Features::Dense(Array1::uninitialized(0)))
                }.combine(pds_other, |x, y| x + y);

                ::std::mem::swap(&mut local_pds, pds);
            } else {
                self.features.insert(c, pds_other.clone());
            }
        }
    }
}

impl Mul<f64> for LFAGradient {
    type Output = LFAGradient;

    fn mul(mut self, factor: f64) -> LFAGradient {
        self.mul_assign(factor);

        self
    }
}

impl MulAssign<f64> for LFAGradient {
    fn mul_assign(&mut self, factor: f64) {
        for pds in self.features.values_mut() {
            pds.mut_activations(|v| v * factor);
        }
    }
}
