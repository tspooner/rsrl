use super::*;
use ndarray::{Array1, Ix1, Ix2};
use std::ops::{Add, AddAssign, Mul, MulAssign};

type GradMap<C> = ::std::collections::HashMap<usize, C>;

#[derive(Clone, Debug, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Columnar<C: Buffer<Dim = Ix1> = Array1<f64>> {
    dim: (usize, usize),
    grads: GradMap<C>,
}

impl<C: Buffer<Dim = Ix1>> Columnar<C> {
    pub fn new(dim: (usize, usize), grads: GradMap<C>) -> Self { Columnar { dim, grads } }

    pub fn from_column(n_cols: usize, index: usize, grad: C) -> Self {
        let dim = (grad.dim(), n_cols);
        let mut grads = GradMap::new();

        grads.insert(index, grad);

        Self::new(dim, grads)
    }
}

impl<C: Buffer<Dim = Ix1>> Buffer for Columnar<C> {
    type Dim = Ix2;

    fn dim(&self) -> (usize, usize) { self.dim }

    fn addto<D: DataMut<Elem = f64>>(&self, weights: &mut ArrayBase<D, Ix2>) {
        for (&c, pds) in self.grads.iter() {
            pds.addto(&mut weights.column_mut(c));
        }
    }

    fn scaled_addto<D: DataMut<Elem = f64>>(&self, alpha: f64, weights: &mut ArrayBase<D, Ix2>) {
        for (&c, buf) in self.grads.iter() {
            buf.scaled_addto(alpha, &mut weights.column_mut(c));
        }
    }

    fn to_dense(&self) -> Array2<f64> {
        let mut g_matrix = Array2::zeros(self.dim);

        self.addto(&mut g_matrix);

        g_matrix
    }
}

impl<C: BufferMut<Dim = Ix1>> BufferMut for Columnar<C> {
    fn zeros(dim: (usize, usize)) -> Self { Self::new(dim, GradMap::new()) }

    fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Columnar {
            dim: self.dim,
            grads: self.grads.iter().map(|(c, j)| (*c, j.map(&f))).collect(),
        }
    }

    fn map_into(mut self, f: impl Fn(f64) -> f64) -> Self {
        self.map_inplace(f);
        self
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.grads.values_mut().for_each(|buf| buf.map_inplace(&f));
    }

    fn merge(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        Columnar {
            dim: self.dim,
            grads: self
                .grads
                .iter()
                .map(|(c, j1)| match other.grads.get(c) {
                    Some(j2) => (*c, j1.merge(j2, &f)),
                    None => (*c, j1.map(|x| f(x, 0.0))),
                })
                .collect(),
        }
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        for (c, xs) in self.grads.iter_mut() {
            if let Some(ys) = other.grads.get(c) {
                xs.merge_inplace(ys, &f)
            } else {
                xs.map_inplace(|x| f(x, 0.0))
            }
        }

        for (&c, ys) in other.grads.iter() {
            self.grads.entry(c).or_insert_with(|| ys.map(|y| f(0.0, y)));
        }
    }
}

impl<C: Buffer<Dim = Ix1>> Into<Array2<f64>> for Columnar<C> {
    fn into(self) -> Array2<f64> { self.to_dense() }
}

impl<C: BufferMut<Dim = Ix1>> Add<Columnar<C>> for Columnar<C> {
    type Output = Columnar<C>;

    fn add(mut self, other: Columnar<C>) -> Columnar<C> {
        self.add_assign(&other);

        self
    }
}

impl<C: BufferMut<Dim = Ix1>> AddAssign<Columnar<C>> for Columnar<C> {
    fn add_assign(&mut self, other: Columnar<C>) {
        for (c, buf_other) in other.grads.into_iter() {
            if let Some(buf) = self.grads.get_mut(&c) {
                buf.merge_inplace(&buf_other, |x, y| x + y);
            } else {
                self.grads.insert(c, buf_other);
            }
        }
    }
}

impl<C: BufferMut<Dim = Ix1>> AddAssign<&Columnar<C>> for Columnar<C> {
    fn add_assign(&mut self, other: &Columnar<C>) {
        for (&c, buf_other) in other.grads.iter() {
            if let Some(buf) = self.grads.get_mut(&c) {
                buf.merge_inplace(buf_other, |x, y| x + y);
            } else {
                self.grads.insert(c, buf_other.to_owned());
            }
        }
    }
}

impl<C: BufferMut<Dim = Ix1>> Mul<f64> for Columnar<C> {
    type Output = Columnar<C>;

    fn mul(mut self, alpha: f64) -> Columnar<C> {
        self.mul_assign(alpha);

        self
    }
}

impl<C: BufferMut<Dim = Ix1>> MulAssign<f64> for Columnar<C> {
    fn mul_assign(&mut self, alpha: f64) {
        for buf in self.grads.values_mut() {
            buf.map_inplace(|x| alpha * x);
        }
    }
}
