use std::f64;
use std::ops::{Add, Sub, Mul, Div};


pub enum Parameter<T: Add + Sub + Mul + Clone + Copy> {
    Fixed(T),
    Exponential {
        value: T,
        floor: T,
        decay: T,
    },
}

impl<T: Add + Sub + Mul + Div + Clone + Copy> Parameter<T> {
    fn value(&self) -> T {
        match self {
            &Parameter::Fixed(v) => v,
            &Parameter::Exponential { value: v, .. } => v,
        }
    }
}

impl Parameter<f64> {
    fn step(self) -> Self {
        match self {
            Parameter::Fixed(v) => self,
            Parameter::Exponential { value: v, floor: f, decay: d } =>
                Parameter::Exponential {
                    value: f64::max(v * d, f),
                    floor: f,
                    decay: d,
                },
        }
    }
}
