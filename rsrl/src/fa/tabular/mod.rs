use ndarray::Array2;
use std::ops::Index;

pub type DenseQTable = Table<Array2<f64>>;

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Table<T>(T);

impl<T, I> Index<I> for Table<T>
where T: Index<I>
{
    type Output = T::Output;

    fn index(&self, index: I) -> &T::Output { self.0.index(index) }
}

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Response;

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Error;

mod dense;
pub use self::dense::*;

mod sparse;
pub use self::sparse::*;
