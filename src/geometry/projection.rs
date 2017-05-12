use super::dimensions::*;

pub trait Projection<D: Dimension> {
    fn project(self, d: &D) -> D::Value;
}

pub fn project<T, D: Dimension>(val: T, d: &D) -> D::Value
    where T: Projection<D>
{
    val.project(d)
}

macro_rules! impl_project {
    ($dt:ty, $st:ident, $dim:pat, $code:block, $($ft:ty),*) => {$(
        impl Projection<$dt> for $ft {
            fn project($st, $dim: &$dt) -> <$dt as Dimension>::Value $code
        }
    )*}
}

impl_project!(Null, self, _, {
    ()
}, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

impl_project!(Infinite, self, _, {
    self as f64
}, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32);

impl Projection<Infinite> for f64 {
    fn project(self, _: &Infinite) -> f64 {
        self
    }
}

impl_project!(Continuous, self, _, {
    self as f64
}, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32);

impl Projection<Continuous> for f64 {
    fn project(self, _: &Continuous) -> f64 {
        self
    }
}

impl_project!(Partitioned, self, d, {
    d.to_partition(self as f64)
}, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32);

impl Projection<Partitioned> for f64 {
    fn project(self, d: &Partitioned) -> usize {
        d.to_partition(self)
    }
}
