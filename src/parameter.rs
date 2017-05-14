use std::f64;
use std::ops::{Add, Sub, Mul, Div};


#[derive(Clone, Copy)]
pub enum Parameter<T: Add + Sub + Mul + Clone + Copy> {
    Fixed(T),
    Exponential {
        value: T,
        floor: T,
        decay: T,
    },
}

impl<T: Add + Sub + Mul + Div + Clone + Copy> Parameter<T> {
    pub fn value(&self) -> T {
        match self {
            &Parameter::Fixed(v) => v,
            &Parameter::Exponential { value: v, .. } => v,
        }
    }
}

impl Parameter<f64> {
    pub fn step(self) -> Self {
        match self {
            Parameter::Fixed(_) => self,
            Parameter::Exponential { value: v, floor: f, decay: d } =>
                Parameter::Exponential {
                    value: f64::max(v * d, f),
                    floor: f,
                    decay: d,
                },
        }
    }

    pub fn back(self) -> Self {
        match self {
            Parameter::Fixed(_) => self,
            Parameter::Exponential { value: v, floor: f, decay: d } =>
                Parameter::Exponential {
                    value: f64::max(v / d, f),
                    floor: f,
                    decay: d,
                },
        }
    }
}

impl Into<Parameter<f64>> for f64 {
    fn into(self) -> Parameter<f64> {
        Parameter::Fixed(self)
    }
}


macro_rules! impl_op {
    ($name: ident, $num_type: ty, $fn_name: ident, $op: tt) => {
        impl $name<$num_type> for Parameter<$num_type> {
            type Output = $num_type;

            fn $fn_name(self, other: $num_type) -> $num_type {
                self.value() $op other
            }
        }

        impl $name<Parameter<$num_type>> for $num_type {
            type Output = $num_type;

            fn $fn_name(self, other: Parameter<$num_type>) -> $num_type {
                self $op other.value()
            }
        }

        impl $name<Parameter<$num_type>> for Parameter<$num_type> {
            type Output = $num_type;

            fn $fn_name(self, other: Parameter<$num_type>) -> $num_type {
                self.value() $op other.value()
            }
        }
    }
}

macro_rules! impl_ops {
    ($($num_type: ty),*) => {$(
        impl_op!(Add, $num_type, add, +);
        impl_op!(Sub, $num_type, sub, -);
        impl_op!(Mul, $num_type, mul, *);
        impl_op!(Div, $num_type, div, /);
    )*}
}

impl_ops!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);


// TODO: Add tests
