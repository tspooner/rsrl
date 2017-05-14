use std::f64;
use std::ops::{Add, Sub, Mul, Div};


#[derive(Clone, Copy)]
pub enum Parameter {
    Fixed(f64),
    Exponential {
        init: f64,
        floor: f64,
        count: u32,
        decay: f64,
    },
    Polynomial {
        init: f64,
        floor: f64,
        count: u32,
        exponent: f64,
    }
}

impl Parameter {
    pub fn fixed(value: f64) -> Self {
        Parameter::Fixed(value)
    }

    pub fn exponential(init: f64, decay: f64, floor: f64) -> Self {
        Parameter::Exponential {
            init: init,
            floor: floor,
            count: 0,
            decay: decay,
        }
    }

    pub fn polynomial(init: f64, exponent: f64, floor: f64) -> Self {
        Parameter::Polynomial {
            init: init,
            floor: floor,
            count: 0,
            exponent: exponent,
        }
    }

    pub fn value(&self) -> f64 {
        match self {
            &Parameter::Fixed(v) => v,

            &Parameter::Exponential {
                init: i, floor: f, count: c, decay: d
            } => f64::max(i * d.powf(c as f64), f),

            &Parameter::Polynomial {
                init: i, floor: f, count: c, exponent: e
            } => f64::max(i * (c as f64).powf(e), f),
        }
    }

    pub fn step(self) -> Self {
        match self {
            Parameter::Fixed(_) => self,
            Parameter::Exponential {
                init: i, floor: f, count: c, decay: d
            } => Parameter::Exponential {
                init: i,
                floor: f,
                count: c + 1,
                decay: d,
            },
            Parameter::Polynomial {
                init: i, floor: f, count: c, exponent: e
            } => Parameter::Polynomial {
                init: i,
                floor: f,
                count: c + 1,
                exponent: e,
            },
        }
    }

    pub fn back(self) -> Self {
        match self {
            Parameter::Fixed(_) => self,
            Parameter::Exponential {
                init: i, floor: f, count: c, decay: d
            } => Parameter::Exponential {
                init: i,
                floor: f,
                count: c - 1,
                decay: d,
            },
            Parameter::Polynomial {
                init: i, floor: f, count: c, exponent: e
            } => Parameter::Polynomial {
                init: i,
                floor: f,
                count: c - 1,
                exponent: e,
            },
        }
    }
}

impl Into<Parameter> for f64 {
    fn into(self) -> Parameter {
        Parameter::Fixed(self)
    }
}


macro_rules! impl_op {
    ($name: ident, $num_type: ty, $fn_name: ident, $op: tt) => {
        impl $name<$num_type> for Parameter {
            type Output = $num_type;

            fn $fn_name(self, other: $num_type) -> $num_type {
                self.value() $op other
            }
        }

        impl $name<Parameter> for $num_type {
            type Output = $num_type;

            fn $fn_name(self, other: Parameter) -> $num_type {
                self $op other.value()
            }
        }

        impl $name<Parameter> for Parameter {
            type Output = $num_type;

            fn $fn_name(self, other: Parameter) -> $num_type {
                self.value() $op other.value()
            }
        }
    }
}

impl_op!(Add, f64, add, +);
impl_op!(Sub, f64, sub, -);
impl_op!(Mul, f64, mul, *);
impl_op!(Div, f64, div, /);


// TODO: Add tests
