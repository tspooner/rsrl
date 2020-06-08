use ndarray::{ArrayBase, DataMut, NdIndex};

#[derive(Copy, Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Tile<D: ndarray::Dimension, I: NdIndex<D>> {
    dim: D::Pattern,
    active: Option<(I, f64)>,
}

impl<D: ndarray::Dimension, I: NdIndex<D>> Tile<D, I> {
    pub fn new(dim: D::Pattern, active: Option<(I, f64)>) -> Self { Tile { dim, active } }
}

impl<D: ndarray::Dimension, I: NdIndex<D> + Clone> crate::params::Buffer for Tile<D, I> {
    type Dim = D;

    fn dim(&self) -> D::Pattern { self.dim.clone() }

    fn addto<E: DataMut<Elem = f64>>(&self, arr: &mut ArrayBase<E, Self::Dim>) {
        if let Some((idx, activation)) = self.active.clone() {
            arr[idx] += activation;
        }
    }

    fn scaled_addto<E: DataMut<Elem = f64>>(&self, alpha: f64, arr: &mut ArrayBase<E, Self::Dim>) {
        if let Some((idx, activation)) = self.active.clone() {
            arr[idx] += alpha * activation;
        }
    }
}

impl<D, I> crate::params::BufferMut for Tile<D, I>
where
    D: ndarray::Dimension,
    I: NdIndex<D> + PartialEq + Clone,
{
    fn zeros(dim: D::Pattern) -> Self { Tile { dim, active: None } }

    fn map(&self, f: impl Fn(f64) -> f64) -> Self { unimplemented!() }

    fn map_into(self, f: impl Fn(f64) -> f64) -> Self {
        Tile {
            dim: self.dim,
            active: self.active.map(|(idx, a)| (idx, f(a))),
        }
    }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) {
        if let Some((_, x)) = &mut self.active {
            *x = f(*x);
        }
    }

    fn merge(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self { unimplemented!() }

    fn merge_into(mut self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        self.merge_inplace(other, f);
        self
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        if self.dim != other.dim {
            panic!("Incompatible buffers shapes.")
        }

        match (&mut self.active, &other.active) {
            (Some((i, x)), Some((j, y))) if i == j => *x = f(*x, *y),
            _ => panic!("Incompatible buffer indices."),
        }
    }
}
