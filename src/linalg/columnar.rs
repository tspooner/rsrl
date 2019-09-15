use crate::{
    geometry::{Vector, VectorView, Matrix, MatrixViewMut},
    linalg::{MatrixLike, Entry},
};
use std::ops::{Add, AddAssign, Mul, MulAssign};

type GradMap = ::std::collections::HashMap<usize, Vector<f64>>;

#[derive(Clone, Debug, Default)]
pub struct Columnar {
    dim: [usize; 2],
    grads: GradMap,
}

impl Columnar {
    pub fn new(n_cols: usize, grads: GradMap) -> Columnar {
        if grads.len() == 0 {
            panic!("No gradient information passed into Columnar::new().");
        } else if grads.len() == 1 {
            Columnar {
                dim: [grads.values().next().unwrap().len(), n_cols],
                grads,
            }
        } else {
            let mut g_iter = grads.values();
            let n_rows = g_iter.next().unwrap().len();

            if g_iter.all(|g| g.len() == n_rows) {
                Columnar {
                    dim: [grads.values().next().unwrap().len(), n_cols],
                    grads,
                }
            } else {
                panic!("Inconsistent gradient vector lengths in Columnar::new().");
            }
        }
    }

    pub fn from_column(n_cols: usize, column: usize, grad: Vector<f64>) -> Columnar {
        let mut grads = GradMap::new();
        grads.insert(column, grad);

        unsafe {
            Self::new_unchecked([grads[&column].len(), n_cols], grads)
        }
    }

    pub unsafe fn new_unchecked(dim: [usize; 2], grads: GradMap) -> Columnar {
        Columnar { dim, grads }
    }
}

impl MatrixLike for Columnar {
    fn zeros(dim: [usize; 2]) -> Self {
        unsafe { Self::new_unchecked(dim, GradMap::new()) }
    }

    fn dim(&self) -> [usize; 2] { self.dim }

    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Columnar {
            dim: self.dim,
            grads: self.grads.into_iter().map(|(c, pds)| (c, pds.mapv(&f))).collect(),
        }
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        self.grads.values_mut().for_each(|pds| pds.iter_mut().for_each(|pd| *pd = f(*pd)));
    }

    fn combine_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        for (c, xs) in self.grads.iter_mut() {
            if let Some(ys) = other.grads.get(c) {
                xs.iter_mut().zip(ys.iter()).for_each(|(x, y)| *x = f(*x, *y));
            } else {
                xs.iter_mut().for_each(|x| *x = f(*x, 0.0));
            }
        }

        for (&c, ys) in other.grads.iter() {
            self.grads.entry(c).or_insert_with(|| ys.iter().map(|y| f(0.0, *y)).collect());
        }
    }

    fn for_each(&self, mut f: impl FnMut(Entry)) {
        for (&c, pds) in self.grads.iter() {
            pds.iter().enumerate().map(|(row, gradient)| Entry {
                index: [row, c],
                gradient: *gradient,
            }).for_each(|pd| f(pd));
        }
    }

    fn addto(&self, weights: &mut MatrixViewMut<f64>) {
        for (&c, pds) in self.grads.iter() {
            let view = unsafe { VectorView::from_shape_ptr(pds.len(), pds.as_ptr()) };

            weights.column_mut(c).add_assign(&view);
        }
    }

    fn scaled_addto(&self, alpha: f64, weights: &mut MatrixViewMut<f64>) {
        for (&c, pds) in self.grads.iter() {
            let view = unsafe { VectorView::from_shape_ptr(pds.len(), pds.as_ptr()) };

            weights.column_mut(c).scaled_add(alpha, &view);
        }
    }
}

impl Into<Matrix<f64>> for Columnar {
    fn into(self) -> Matrix<f64> {
        let mut g_matrix = Matrix::zeros((self.dim[0], self.dim[1]));

        for (c, g_vector) in self.grads.into_iter() {
            let view = unsafe { VectorView::from_shape_ptr(g_vector.len(), g_vector.as_ptr()) };

            g_matrix.column_mut(c).assign(&view);
        }

        g_matrix
    }
}

impl Add<Columnar> for Columnar {
    type Output = Columnar;

    fn add(mut self, other: Columnar) -> Columnar {
        self.add_assign(&other);

        self
    }
}

impl AddAssign<Columnar> for Columnar {
    fn add_assign(&mut self, other: Columnar) {
        for (c, pds_other) in other.grads.into_iter() {
            if let Some(pds) = self.grads.get_mut(&c) {
                pds.iter_mut().zip(pds_other.iter()).for_each(|(x, y)| *x += y);
            } else {
                self.grads.insert(c, pds_other);
            }
        }
    }
}

impl AddAssign<&Columnar> for Columnar {
    fn add_assign(&mut self, other: &Columnar) {
        for (&c, pds_other) in other.grads.iter() {
            if let Some(pds) = self.grads.get_mut(&c) {
                pds.iter_mut().zip(pds_other.iter()).for_each(|(x, y)| *x += y);
            } else {
                self.grads.insert(c, pds_other.clone());
            }
        }
    }
}

impl Mul<f64> for Columnar {
    type Output = Columnar;

    fn mul(mut self, factor: f64) -> Columnar {
        self.mul_assign(factor);

        self
    }
}

impl MulAssign<f64> for Columnar {
    fn mul_assign(&mut self, factor: f64) {
        for pds in self.grads.values_mut() {
            pds.iter_mut().for_each(|x| *x *= factor);
        }
    }
}
